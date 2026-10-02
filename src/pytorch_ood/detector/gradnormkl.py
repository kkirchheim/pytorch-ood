"""

..  autoclass:: pytorch_ood.detector.GradNormKL
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""

from typing import Callable, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import DetectorInfo, GradientDetector, ModelNotSetException, Paper, Task
from .gradnorm import _requires_grad

try:
    from torch.func import (
        functional_call as _functional_call,
    )
    from torch.func import (
        grad as _func_grad,
    )
    from torch.func import (
        vmap as _vmap,
    )

    _TORCH_FUNC_AVAILABLE = True
except ImportError:
    _TORCH_FUNC_AVAILABLE = False


def _uniform_cross_entropy(logits: Tensor, temperature: float) -> Tensor:
    # cross-entropy to the uniform distribution, which is the KL divergence up to the constant
    # log C; the official code sums instead of averaging over the classes, which scales all
    # scores by C and does not change their ranking
    return -(logits / temperature).log_softmax(dim=1).mean()


class GradNormKL(GradientDetector):
    """
    Detector from the paper *On the Importance of Gradients for Detecting Distributional Shifts
    in the Wild*.

    For each input sample, computes the KL divergence between the uniform distribution
    :math:`u` and the softmax output at temperature :math:`T`, implemented as the cross-entropy to
    :math:`u`, which differs from the KL divergence only by a constant:

    .. math::
        \\mathcal{L}(x) = -\\frac{1}{C} \\sum_{c=1}^{C} \\log \\frac{e^{f_c(x) / T}}{\\sum_{c'} e^{f_{c'}(x) / T}}

    The outlier score is the :math:`\\ell_1`-norm of the gradients of this loss w.r.t. the selected
    parameters :math:`w` of the model, negated relative to the paper:

    .. math::
        -\\left\\lVert \\frac{\\partial \\mathcal{L}(x)}{\\partial w} \\right\\rVert_1

    The gradient w.r.t. the logits is :math:`(\\text{softmax}(f(x) / T) - u) / T`, which is zero
    when the model predicts a uniform distribution and grows as the prediction becomes more
    peaked. In-distribution inputs typically get more confident predictions, and thus larger
    gradient norms, than OOD inputs.

    .. note:: The paper uses only the gradients of the weights of the final classification layer.
        You can achieve this by setting ``param_filter``, e.g. ``lambda name: name == "fc.weight"``.
        Gradients are only computed for the selected parameters.

    .. note:: On PyTorch ≥ 2.0, per-sample gradients are computed with ``torch.func.vmap`` +
        ``torch.func.grad`` in a single batched forward+backward pass. On PyTorch 1.x the
        original sequential loop over individual samples is used as a fallback.
    """

    info = DetectorInfo(
        paper=Paper(
            title="On the Importance of Gradients for Detecting Distributional Shifts in the Wild",
            venue="NeurIPS",
            year=2021,
            url="https://arxiv.org/abs/2110.00218",
            code="https://github.com/deeplearning-wisc/gradnorm_ood",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    def __init__(
        self,
        model: torch.nn.Module,
        param_filter: Callable[[str], bool] = None,
        micro_batch_size: int = 32,
        temperature: float = 1.0,
    ):
        """
        :param model: A pre-trained classification model :math:`f`.
        :param param_filter: Function indicating whether a named parameter should be included in
            the scoring, selecting the parameters :math:`w`. Gradients are only computed for these.
            If ``None``, all parameters are used.
        :param micro_batch_size: maximum number of samples whose per-sample gradients are computed at
            once. Smaller values reduce peak memory. Must be at least 1. Only used on PyTorch >= 2.0.
        :param temperature: temperature :math:`T` of the softmax
        :raises ModelNotSetException: if ``model`` is ``None``
        """
        # _predict_batched splits the input into chunks of at most micro_batch_size before calling vmap:
        # per-sample gradients require holding every sample's forward activations through the full network
        # simultaneously, so peak memory scales linearly with batch size (e.g. ~53 GB at
        # batch size 128 for a ResNet-50 at 224x224, even restricted to a single layer's
        # gradients). Chunking bounds peak memory independent of the caller's batch size.
        if model is None:
            raise ModelNotSetException("Model must be provided.")

        def default_filter(x):
            return True

        self.param_filter = param_filter or default_filter
        self.model = model
        self.micro_batch_size = micro_batch_size
        self.temperature = temperature

    def fit(self, data_loader: DataLoader, **kwargs) -> Self:
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Compute outlier scores for an input batch.

        :param x: input tensor of shape :math:`B \\times \\ldots`, will be passed through the network
        :return: outlier scores of shape :math:`B`
        :raises ValueError: if ``param_filter`` selects no parameter
        """
        if self.model is None:
            raise ModelNotSetException()

        x = x.to(self.device)

        if _TORCH_FUNC_AVAILABLE:
            return self._predict_batched(x)
        return self._predict_sequential(x)

    def _selected_params(self) -> Dict[str, Tensor]:
        selected = {n: p for n, p in self.model.named_parameters() if self.param_filter(n)}
        if not selected:
            raise ValueError("param_filter selects none of the parameters of the model")
        return selected

    def _predict_batched(self, x: Tensor) -> Tensor:
        """Vectorized per-sample gradients via torch.func (PyTorch ≥ 2.0), chunked to
        bound peak memory (see ``micro_batch_size``)."""
        selected = self._selected_params()
        # torch.func.grad differentiates w.r.t. everything it is given, regardless of
        # requires_grad, so only the selected parameters are passed as its argument
        constants = {n: p for n, p in self.model.named_parameters() if n not in selected}
        constants.update(self.model.named_buffers())
        model = self.model
        temperature = self.temperature

        def loss_for_single(params, x_single):
            logits = _functional_call(model, {**params, **constants}, (x_single.unsqueeze(0),))
            return _uniform_cross_entropy(logits, temperature)

        chunks = []
        with torch.enable_grad():
            for start in range(0, x.shape[0], self.micro_batch_size):
                x_chunk = x[start : start + self.micro_batch_size]
                per_sample_grads = _vmap(_func_grad(loss_for_single), in_dims=(None, 0))(
                    selected, x_chunk
                )
                chunks.append(
                    sum(
                        g.abs().sum(dim=tuple(range(1, g.ndim))) for g in per_sample_grads.values()
                    )
                )

        return -torch.cat(chunks).detach()

    def _predict_sequential(self, x: Tensor) -> Tensor:
        """Per-sample gradients via serial backward passes (PyTorch < 2.0 fallback)."""
        selected = list(self._selected_params().values())
        scores = []

        with _requires_grad(selected):
            for xi in x:
                with torch.enable_grad():
                    loss = _uniform_cross_entropy(self.model(xi.unsqueeze(0)), self.temperature)
                    # only the selected parameters, without writing .grad into the model
                    grads = torch.autograd.grad(loss, selected)
                scores.append(-sum(g.abs().sum() for g in grads))

        return torch.stack(scores).detach()
