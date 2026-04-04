"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.GradNormKL
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing import TypeVar, Callable

from ..api import Detector, ModelNotSetException

try:
    from torch.func import grad as _func_grad, vmap as _vmap, functional_call as _functional_call

    _TORCH_FUNC_AVAILABLE = True
except ImportError:
    _TORCH_FUNC_AVAILABLE = False

Self = TypeVar("Self")


class GradNormKL(Detector):
    """
    Detector from the paper *On the Importance of Gradients for Detecting Distributional Shifts
    in the Wild*.

    For each input sample, computes the KL divergence between the softmax output and a uniform
    distribution (implemented via binary cross-entropy with a uniform confounding label of
    :math:`1/C` per class). The outlier score is the **negated** :math:`\\ell_1`-norm of the
    gradients of this loss w.r.t. the selected model parameters.

    The key insight is that the gradient w.r.t. the logits simplifies to
    :math:`\\text{softmax}(z) - 1/C`, which is zero when the model predicts a uniform distribution
    and grows as the prediction becomes more peaked. For in-distribution inputs the model is
    typically more confident (larger gradient norm) than for OOD inputs, so the negated norm gives
    higher scores to OOD samples, consistent with the convention that higher outlier scores
    indicate OOD data.

    .. note:: The paper recommends using only the gradients of the final classification head
        (last FC layer) for computational efficiency. You can achieve this by setting
        ``param_filter`` and disabling gradient computation for the backbone via
        ``model.requires_grad_(False); model.fc.requires_grad_(True)``.

    .. note:: On PyTorch ≥ 2.0, per-sample gradients are computed with ``torch.func.vmap`` +
        ``torch.func.grad`` in a single batched forward+backward pass. On PyTorch 1.x the
        original sequential loop over individual samples is used as a fallback.

    :see Paper: `NeurIPS <https://arxiv.org/abs/2110.00218>`__
    """

    def __init__(self, model: torch.nn.Module, param_filter: Callable[[str], bool] = None):
        """
        :param model: A pre-trained classification model.
        :param param_filter: Function indicating whether a named parameter should be included in
            the scoring. If ``None``, all parameters are used.
        """
        if model is None:
            raise ModelNotSetException("Model must be provided.")

        def default_filter(x):
            return True

        self.param_filter = param_filter or default_filter
        self.model = model

    def fit(self, data_loader: DataLoader, **kwargs) -> Self:
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Compute outlier scores for an input batch.

        Uses the device of the model parameters for all computations.
        On PyTorch ≥ 2.0, per-sample gradients are batched via ``torch.func``; on older
        versions a sequential loop is used.

        :param x: input tensor, will be passed through the network
        :return: vector of outlier scores (higher = more likely OOD)
        """
        if self.model is None:
            raise ModelNotSetException()

        device = next(self.model.parameters()).device
        x = x.to(device)

        if _TORCH_FUNC_AVAILABLE:
            return self._predict_batched(x)
        return self._predict_sequential(x)

    def _predict_batched(self, x: Tensor) -> Tensor:
        """Vectorized per-sample gradients via torch.func (PyTorch ≥ 2.0)."""
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        model = self.model
        param_filter = self.param_filter

        def loss_for_single(params, x_single):
            logits = _functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
            C = logits.shape[1]
            y_uniform = torch.ones_like(logits) / C
            return F.binary_cross_entropy(logits.softmax(dim=1), y_uniform, reduction="sum")

        with torch.enable_grad():
            per_sample_grads = _vmap(_func_grad(loss_for_single), in_dims=(None, 0))(params, x)

        total_norms = x.new_zeros(x.shape[0])
        for name, g in per_sample_grads.items():
            if param_filter(name):
                total_norms = total_norms + g.abs().sum(dim=tuple(range(1, g.ndim)))

        return -total_norms

    def _predict_sequential(self, x: Tensor) -> Tensor:
        """Per-sample gradients via serial backward passes (PyTorch < 2.0 fallback)."""
        device = x.device
        scores = []

        for xi in x:
            with torch.enable_grad():
                self.model.zero_grad()
                logits = self.model(xi.unsqueeze(0))
                C = logits.shape[1]
                y_uniform = torch.ones_like(logits) / C
                loss = F.binary_cross_entropy(logits.softmax(dim=1), y_uniform, reduction="sum")
                loss.backward()

                total_norm = torch.tensor(0.0, device=device)
                for name, p in self.model.named_parameters():
                    if self.param_filter(name) and p.grad is not None:
                        total_norm = total_norm + p.grad.detach().abs().sum()

                scores.append(-total_norm)

        return torch.stack(scores)
