"""

..  autoclass:: pytorch_ood.detector.GradNorm
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""

from contextlib import contextmanager
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import DetectorInfo, GradientDetector, ModelNotSetException, Paper, Task

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


@contextmanager
def _requires_grad(params: List[Tensor]):
    """Temporarily enables gradients for ``params``, so that frozen parameters can be scored."""
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


class GradNorm(GradientDetector):
    """
    Detector from the paper *Gradients as a Measure of Uncertainty in Neural Networks*.

    For each input sample, computes the binary cross-entropy loss between the softmax output and a "confounding
    label", which is a vector of all ones. Then, for each set of parameters in the model (as given
    by ``model.named_parameters()``), computes the squared :math:`\\ell_2`-norm of the
    gradients of the loss w.r.t. that parameter. The outlier score is the negative sum of these squared norms.

    The gradient of this loss w.r.t. the logits is :math:`C p - 1` for :math:`C` classes and softmax output
    :math:`p`. It vanishes for a uniform softmax and is large for a confident prediction, so the gradient norm is
    larger for inputs the model is familiar with. The sign is therefore flipped, so that uncertain inputs receive
    higher outlier scores.

    .. note:: Using only the gradients of the final classification head makes this computationally cheaper.
     You can achieve this by setting ``param_filter``; gradients are only computed for the selected
     parameters. For an example, see the :doc:`GradNorm example </auto_examples/detectors/gradnorm>`.

    The model is not switched to evaluation mode, so layers such as batch normalization and dropout behave
    according to ``model.training``; you should usually call ``model.eval()`` first. Gradients are computed per
    sample, and :meth:`predict` must not be wrapped in ``torch.no_grad()``.

    .. note:: On PyTorch ≥ 2.0, per-sample gradients are computed with ``torch.func.vmap`` +
        ``torch.func.grad`` in a single batched forward+backward pass. On PyTorch 1.x the
        original sequential loop over individual samples is used as a fallback.

    .. warning::
        This implementation sums the norms of all (selected) parameters into a single scalar and uses its negative
        directly as an outlier score, without any training. This requires no OOD data, but may perform poorly when ID and
        OOD datasets are of similar complexity. For an unsupervised
        gradient-based alternative see :class:`~pytorch_ood.detector.GradNormKL`.
    """

    # The paper's actual experiments (Section 4) concatenate the per-layer squared L2 norms into a feature vector
    # and then train a 2-layer FC binary classifier on labeled ID and OOD gradient representations, which learns
    # the direction of the score. This class is a significant simplification: it uses the scalar sum, negated
    # because the gradient norm is largest for confident predictions.
    # OpenOOD uses only the gradients of the final classification head.

    info = DetectorInfo(
        paper=Paper(
            title="Gradients as a Measure of Uncertainty in Neural Networks",
            venue="ICIP",
            year=2020,
            url="https://arxiv.org/abs/2008.08030v2",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    def __init__(self, model: torch.nn.Module, param_filter: Callable[[str], bool] = None):
        """
        :param model: A pre-trained classification model
        :param param_filter: Function which indicates whether a named parameter should be included in the scoring.
            Gradients are only computed for these. If ``None``, all parameters are used.
        :raises ModelNotSetException: if ``model`` is ``None``
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

        :param x: input of shape :math:`B \\times \\ldots`, will be passed through the network
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
        """Vectorized per-sample gradients via torch.func (PyTorch ≥ 2.0)."""
        selected = self._selected_params()
        # torch.func.grad differentiates w.r.t. everything it is given, regardless of
        # requires_grad, so only the selected parameters are passed as its argument
        constants = {n: p for n, p in self.model.named_parameters() if n not in selected}
        constants.update(self.model.named_buffers())
        model = self.model

        def loss_for_single(params, x_single):
            logits = _functional_call(model, {**params, **constants}, (x_single.unsqueeze(0),))
            y_conf = torch.ones_like(logits)
            return F.binary_cross_entropy(logits.softmax(dim=1), y_conf, reduction="sum")

        with torch.enable_grad():
            per_sample_grads = _vmap(_func_grad(loss_for_single), in_dims=(None, 0))(selected, x)

        return -sum(
            (g**2).sum(dim=tuple(range(1, g.ndim))) for g in per_sample_grads.values()
        ).detach()

    def _predict_sequential(self, x: Tensor) -> Tensor:
        """Per-sample gradients via serial backward passes (PyTorch < 2.0 fallback)."""
        selected = list(self._selected_params().values())
        scores = []

        with _requires_grad(selected):
            for xi in x:
                with torch.enable_grad():
                    logits = self.model(xi.unsqueeze(0))
                    y_conf = torch.ones_like(logits)
                    loss = F.binary_cross_entropy(logits.softmax(dim=1), y_conf, reduction="sum")
                    # only the selected parameters, without writing .grad into the model
                    grads = torch.autograd.grad(loss, selected)
                scores.append(sum((g**2).sum() for g in grads))

        return -torch.stack(scores).detach()
