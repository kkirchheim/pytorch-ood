"""

..  autoclass:: pytorch_ood.detector.TemperatureScaling
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Optional

import torch.nn
from torch import Tensor, tensor
from torch.nn import Module
from torch.nn.functional import log_softmax, nll_loss
from torch.optim import LBFGS
from typing_extensions import Self

from pytorch_ood.detector.softmax import MaxSoftmax
from pytorch_ood.utils import is_known

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
        super(TemperatureScaling, self).__init__(model=model)
        self.t = torch.nn.Parameter(tensor(1.0))
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
        Optimize the temperature by minimizing the negative log-likelihood using L-BFGS
        (50 iterations, so the optimum is only approximately reached). Ignores OOD inputs.

        :param logits: logits of shape :math:`N \\times C`
        :param labels: labels of shape :math:`N`
        :return: self
        :raise ValueError: if there are no ID samples
        """
        known = is_known(labels)

        if not known.any():
            raise ValueError("No ID samples")

        optimizer = LBFGS([self.t], lr=0.01, max_iter=50)

        device = self.t.device

        logits = logits[known].to(device)
        labels = labels[known].to(device)

        with torch.no_grad():
            loss = nll_loss(log_softmax(logits / self.t, dim=1), labels).item()

        log.info(f"Initial T/NLL: {self.t.item():.3f}/{loss:.3f}")

        def closure():
            optimizer.zero_grad()
            loss = nll_loss(log_softmax(logits / self.t, dim=1), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            loss = nll_loss(log_softmax(logits / self.t, dim=1), labels).item()

        log.info(f"Optimal temperature: {self.t.item()}")
        log.info(f"NLL after scaling: {loss:.2f}'")

        self._is_fitted = True

        return self
