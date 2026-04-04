"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.GradNorm
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


class GradNorm(Detector):
    """
    Detector from the paper *Gradients as a Measure of Uncertainty in Neural Networks*.

    For each input sample, computes the binary cross-entropy loss between logits and a "confounding label",
    which is a vector of all ones. Then, for each set of parameters in the model (as given
    by ``model.named_parameters()``), computes up the squared :math:`\\ell_2`-norm of the
    gradients of the loss w.r.t. that parameter. The outlier score is the sum of these squared norms.

    The idea is that higher gradient norms indicates that the model would require large
    parameter updates to accommodate the input, i.e., for such data, it is less familiar or
    more uncertain, and hence more likely to be OOD.

    .. note:: OpenOOD uses only the gradients of the final classification head, which
     makes this computationally cheaper. You can achieve something similar by setting ``param_filter``. Still, this
     method will compute gradients for all parameters unless you explicitly deactivate
     gradient calculation for parameters. For an example, see :doc:`here <auto_examples/detectors/gradnorm>`

    .. note:: On PyTorch ≥ 2.0, per-sample gradients are computed with ``torch.func.vmap`` +
        ``torch.func.grad`` in a single batched forward+backward pass. On PyTorch 1.x the
        original sequential loop over individual samples is used as a fallback.

    .. warning::
        The paper's actual experiments (Section 4) concatenate the per-layer squared L2 norms into a
        feature vector and then **train a 2-layer FC binary classifier** on labeled ID and OOD
        gradient representations. The current implementation is a significant simplification: it
        sums all norms into a single scalar and uses it as a direct outlier score without any
        training. This simplification requires no OOD data but tends to perform poorly (AUROC ≈ 0.5)
        when ID and OOD datasets are of similar complexity, because the scalar sum loses the
        per-layer discriminative structure the classifier exploits. For an unsupervised
        gradient-based alternative see :class:`~pytorch_ood.detector.GradNormKL`.

    :see Paper: `ICIP <https://arxiv.org/abs/2008.08030v2>`__
    """

    def __init__(self, model: torch.nn.Module, param_filter: Callable[[str], bool] = None):
        """
        :param model: A pre-trained classification model
        :param param_filter: Function which indicates whether a named parameter should be included in the scoring. If none
            give, all parameters will be used.
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
        Compute outlier scores from input batch.

        We will use the device of the model parameters for computations.
        On PyTorch ≥ 2.0, per-sample gradients are batched via ``torch.func``; on older
        versions a sequential loop is used.

        :param x: input, will be passed through network
        :return: vector of outlier scores
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
            y_conf = torch.ones_like(logits)
            return F.binary_cross_entropy(logits.softmax(dim=1), y_conf, reduction="sum")

        with torch.enable_grad():
            per_sample_grads = _vmap(_func_grad(loss_for_single), in_dims=(None, 0))(params, x)

        total_norms = x.new_zeros(x.shape[0])
        for name, g in per_sample_grads.items():
            if param_filter(name):
                total_norms = total_norms + (g**2).sum(dim=tuple(range(1, g.ndim)))

        return total_norms

    def _predict_sequential(self, x: Tensor) -> Tensor:
        """Per-sample gradients via serial backward passes (PyTorch < 2.0 fallback)."""
        device = x.device
        scores = []

        for xi in x:
            with torch.enable_grad():
                self.model.zero_grad()
                logits = self.model(xi.unsqueeze(0))
                y_conf = torch.ones_like(logits, device=device)
                loss = F.binary_cross_entropy(logits.softmax(dim=1), y_conf, reduction="sum")
                loss.backward()

                total_norm = torch.tensor(0.0, device=device)
                for name, p in self.model.named_parameters():
                    if self.param_filter(name) and p.grad is not None:
                        total_norm = total_norm + torch.sum(p.grad.detach() ** 2)

                scores.append(total_norm)

        return torch.stack(scores)
