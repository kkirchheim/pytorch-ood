"""
.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.NACUE
    :members:
    :exclude-members:
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from pytorch_ood.api import Detector
from pytorch_ood.api import ModelNotSetException, RequiresFittingException


def _default_feature_reduce(z: Tensor) -> Tensor:
    """
    Reduce a layer output `z` to a 2D tensor (B, N).
    - If z is already (B, N): leave as is.
    - If z is (B, C, H, W): average over spatial dims => (B, C)
    - Otherwise: flatten all but batch => (B, -1)
    """
    if z.ndim == 2:
        return z
    if z.ndim == 4:
        return z.mean(dim=(2, 3))
    return z.flatten(start_dim=1)


def _bin_indices(z_hat: Tensor, m: int) -> Tensor:
    """
    Map z_hat in [0,1] to integer bins in {0,...,m-1}.
    """
    z_hat = torch.clamp(z_hat, 0.0, 1.0)
    # edge case: z_hat == 1.0 -> m, clamp to m-1
    idx = torch.floor(z_hat * m).to(torch.long)
    return torch.clamp(idx, 0, m - 1)


def _bin_indices_logit(z_hat: torch.Tensor, m: int, U: float = 10.0) -> torch.Tensor:
    eps = 1e-6
    z = z_hat.clamp(eps, 1 - eps)
    u = torch.log(z) - torch.log1p(-z)  # logit
    u = u.clamp(-U, U)
    t = (u + U) / (2 * U)  # in [0,1]
    idx = (t * m).floor().clamp(0, m - 1).long()
    return idx


def _bin_indices_geometric(z_hat: torch.Tensor, m: int, eps: float = 1e-6) -> torch.Tensor:
    # symmetric geometric bins on (0, 1): dense near 0 and 1
    z = z_hat.clamp(eps, 1 - eps)

    m1 = m // 2
    m2 = m - m1

    # edges for [eps, 0.5]
    left = torch.logspace(
        torch.log10(torch.tensor(eps, device=z.device)),
        torch.log10(torch.tensor(0.5, device=z.device)),
        steps=m1 + 1,
        device=z.device,
    )
    # edges for (0.5, 1-eps] mirrored
    right = 1.0 - left.flip(0)

    # combine (avoid duplicating 0.5)
    edges = torch.cat([left, right[1:]], dim=0)  # length m+1

    # bucketize gives indices in [0, m]
    idx = torch.bucketize(z, edges, right=False) - 1
    return idx.clamp(0, m - 1).long()


@dataclass
class _LayerStats:
    m: int
    o_star: int
    counts: Tensor  # (N, M) int64


class NACUE(Detector):
    """
    Neuron-Activated Coverage from the paper from the paper *Neuron Activation Coverage: Rethinking out-of-Distribution detection and generalization*

    :see Paper:
        `ICLR <https://arxiv.org/pdf/2306.02879>`__
    """

    def __init__(
        self,
        model: Optional[Module],
        layers: Sequence[Module],
        m_bins: Union[int, Sequence[int]] = 50,
        alpha: Union[float, Sequence[float]] = 100.0,
        o_star: Union[int, Sequence[int]] = 50,
        feature_reduce: Callable[[Tensor], Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        :param model: A classifier that returns logits of shape :math:`(B, C)`, where :math:`B` denotes the batch size and
                      :math:`C` the number of classes.
        :param layers: Sequence of modules whose outputs :math:`z` are used to compute NAC. For a ResNet-style architecture, e.g.
                      ``[model.layer1, model.layer2, model.layer3, model.layer4]``.
        :param m_bins: Number of histogram bins :math:`M`. Either a single value (shared across all layers) or one value per layer.
        :param alpha: Sigmoid steepness parameter :math:`\\alpha`. Either a single value (shared across all layers) or one value
                      per layer.
        :param o_star: Bin-filling parameter :math:`O^*` (minimum count required for full coverage). Either a single value
                       (shared across all layers) or one value per layer.
        :param feature_reduce: Function mapping a layer output tensor to a 2D tensor of shape :math:`(B, N)`, where :math:`B`
                              denotes the batch size and :math:`N` the number of neurons. Defaults to: identity for tensors of
                              shape :math:`(B, N)`, spatial mean for tensors of shape :math:`(B, C, H, W)`, otherwise flatten.
        :param device: Optional device used during fitting and prediction.
        """
        self.model = model
        self.layers = list(layers)
        self.feature_reduce = feature_reduce or _default_feature_reduce
        self.device = torch.device(device) if device is not None else None

        def _expand(v, cast):
            if isinstance(v, (list, tuple)):
                if len(v) != len(self.layers):
                    raise ValueError(
                        "If you pass per-layer hyperparameters, length must match `layers`."
                    )
                return [cast(x) for x in v]
            return [cast(v) for _ in self.layers]

        self.m_bins = _expand(m_bins, int)
        self.alpha = _expand(alpha, float)
        self.o_star = _expand(o_star, int)

        self._fitted: bool = False
        self._stats: List[_LayerStats] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._captured: List[Tensor] = []

    # ----------------------------- pytorch-ood API -----------------------------

    def fit(self, data_loader: DataLoader, device=None) -> "NACUE":
        if self.model is None:
            raise ModelNotSetException("NACUE requires a model.")
        self.model.eval()

        # allow pytorch-ood's fit(..., device=...) convention
        if device is not None:
            self.device = device

        if self.device is not None:
            self.model.to(self.device)

        self._init_hooks()

        # First pass: initialize histograms (need N per layer)
        self._stats = []
        with torch.enable_grad():
            for batch in data_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device) if self.device is not None else x
                self._captured.clear()
                logits = self.model(x)
                _ = logits.sum()

                if not self._stats:
                    for li, z in enumerate(self._captured):
                        z2 = self.feature_reduce(z)
                        n = z2.shape[1]
                        counts = torch.zeros((n, self.m_bins[li]), dtype=torch.long, device="cpu")
                        self._stats.append(
                            _LayerStats(m=self.m_bins[li], o_star=self.o_star[li], counts=counts)
                        )
                break

        # Second pass: accumulate histograms
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(self.device) if self.device is not None else x

            with torch.enable_grad():
                self._captured.clear()
                logits = self.model(x)
                p = torch.softmax(logits, dim=1)
                dkl = -(torch.log(p + 1e-12)).mean(dim=1)
                loss = dkl.sum()

                grads = torch.autograd.grad(
                    loss,
                    self._captured,  # list of tensors
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                for li, (z, grad) in enumerate(zip(self._captured, grads)):
                    # grad = torch.autograd.grad(loss, z, retain_graph=True, create_graph=False)[0]
                    prod = z * grad
                    p2 = self.feature_reduce(prod)  # (B, N)
                    z_hat = torch.sigmoid(self.alpha[li] * p2)

                    m = self._stats[li].m
                    idx = _bin_indices_geometric(z_hat.detach(), m=m)
                    n = idx.shape[1]

                    neuron_offsets = (
                        torch.arange(n, device=idx.device, dtype=torch.long) * m
                    ).view(1, n)
                    index_flat = (neuron_offsets + idx).flatten()

                    ones = torch.ones_like(index_flat, dtype=torch.long)
                    counts_flat = self._stats[li].counts.view(-1)
                    counts_flat.scatter_add_(0, index_flat.to("cpu"), ones.to("cpu"))

        self._fitted = True
        self._remove_hooks()
        return self

    def fit_features(self, x: Tensor, y: Tensor) -> "NACUE":
        raise NotImplementedError(
            "NACUE requires model forward + gradients, so fit_features is not supported."
        )

    def predict(self, x: Tensor) -> Tensor:
        if self.model is None:
            raise ModelNotSetException("NACUE requires a model.")
        if not self._fitted:
            raise RequiresFittingException("Call `fit` before `predict`.")
        self.model.eval()
        if self.device is not None:
            self.model.to(self.device)
            x = x.to(self.device)

        self._init_hooks()

        with torch.enable_grad():
            self._captured.clear()
            logits = self.model(x)
            p = torch.softmax(logits, dim=1)
            dkl = -(torch.log(p + 1e-12)).mean(dim=1)  # (B,)
            loss = dkl.sum()

            layer_scores: List[Tensor] = []
            for li, z in enumerate(self._captured):
                grad = torch.autograd.grad(loss, z, retain_graph=True, create_graph=False)[0]
                prod = z * grad
                p2 = self.feature_reduce(prod)  # (B, N)
                z_hat = torch.sigmoid(self.alpha[li] * p2)

                m = self._stats[li].m
                idx = _bin_indices_geometric(z_hat.detach(), m=m)  # (B, N)

                counts = self._stats[li].counts.to(idx.device)  # (N, M)
                # gather counts for each (sample, neuron)
                # idx_T: (N, B) to gather along dim=1
                idx_t = idx.transpose(0, 1)  # (N, B)
                o = counts.gather(dim=1, index=idx_t).transpose(0, 1).to(torch.float32)  # (B, N)

                phi = torch.clamp(o / float(self._stats[li].o_star), max=1.0)  # (B, N)
                s = phi.mean(dim=1)  # (B,)
                layer_scores.append(s)

            id_likeness = torch.stack(layer_scores, dim=0).sum(dim=0)  # (B,)
            outlier_score = -id_likeness  # pytorch-ood convention: larger => more outlier
        self._remove_hooks()
        return outlier_score.detach()

    def predict_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "NACUE requires model forward + gradients, so predict_features is not supported."
        )

    # ----------------------------- internals -----------------------------

    def _init_hooks(self) -> None:
        if self._hooks:
            return

        self._captured = []

        def _hook(_module: Module, _inp: Tuple[Tensor, ...], out: Tensor):
            # keep tensor in computation graph (no detach)
            self._captured.append(out)

        for layer in self.layers:
            self._hooks.append(layer.register_forward_hook(_hook))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
