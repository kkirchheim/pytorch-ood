"""

..  autoclass:: pytorch_ood.detector.TemperatureScaling
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Optional

import torch.nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax, nll_loss
from torch.optim import LBFGS
from typing_extensions import Self

from pytorch_ood.detector.softmax import MaxSoftmax

from ..api import DetectorInfo, Paper, RequiresFittingException, Task

log = logging.getLogger(__name__)


class TemperatureScaling(MaxSoftmax):
    """
    Implements temperature scaling from the paper
    *On Calibration of Modern Neural Networks*.

    The method uses an additional set of held-out ID validation samples to determine the optimal temperature
    value :math:`T` to calibrate the softmax output. :math:`T` is initialized to 1 and estimated by ``fit``;
    samples with a label below zero are ignored. Calling ``predict`` before fitting raises a
    :class:`~pytorch_ood.api.RequiresFittingException`.

    The score is calculated as:

    .. math:: - \\max_y \\sigma_y(f(x) / T)

    where :math:`\\sigma` is the softmax function, :math:`T` is the optimal temperature and :math:`\\sigma_y`
    indicates the :math:`y^{th}` value of the resulting probability vector.
    """

    info = DetectorInfo(
        paper=Paper(
            title="On Calibration of Modern Neural Networks",
            venue="ICML",
            year=2017,
            url="https://arxiv.org/pdf/1706.04599.pdf",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    def __init__(self, model: Optional[Module]):
        """
        :param model: neural network to use. Can be ``None`` when using
            ``fit_logits(...)`` and ``predict_logits(...)`` directly.
        """
        # t stays the float 1.0 of MaxSoftmax until fit makes it a parameter on the fit device: a
        # CPU tensor created here would come first in vars(self) and decide the detector's device
        super(TemperatureScaling, self).__init__(model=model)
        self._is_fitted = False

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input batch, will be passed through the model
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector was not fitted
        """
        return super().predict(x)

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits of shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector was not fitted
        """
        if not self._is_fitted:
            raise RequiresFittingException()

        return super().predict_logits(logits)

    def fit_logits(self, logits: Tensor, labels: Tensor) -> Self:
        """
        Find the temperature that minimizes the negative log-likelihood of the labels, using L-BFGS.
        Every fit starts again from :math:`T = 1`. Ignores OOD inputs.

        :param logits: logits of shape :math:`N \\times C`
        :param labels: labels of shape :math:`N`
        :return: self
        :raise ValueError: if there are no ID samples
        """
        device = self.device or logits.device
        logits = logits.detach().to(device)
        labels = labels.to(device)

        # optimize log T, which keeps T positive; the strong Wolfe line search makes the step size
        # adaptive, so the optimum is reached for temperatures far from 1
        log_t = torch.zeros((), device=device, requires_grad=True)
        optimizer = LBFGS([log_t], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = nll_loss(log_softmax(logits / log_t.exp(), dim=1), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.t = torch.nn.Parameter(log_t.detach().exp())
        log.info(f"Optimal temperature: {self.t.item():.3f}")
        self._is_fitted = True
        return self
