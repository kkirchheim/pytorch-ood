"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.TemperatureScaling
    :members:

"""
from typing import Optional, TypeVar

import torch.nn
from torch import Tensor, tensor
from torch.nn.functional import nll_loss
from torch.nn import Module
from torch.optim import LBFGS
from torch.utils.data import DataLoader
import logging

from ..api import RequiresFittingException
from pytorch_ood.detector.softmax import MaxSoftmax
from pytorch_ood.utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class TemperatureScaling(MaxSoftmax,  torch.nn.Module):
    """
    Implements temperature scaling from the paper
    *On Calibration of Modern Neural Networks*.

    The method uses an additional set of validation samples to determine the optimal temperature
    value :math:`T` to calibrate the softmax output.

    The score is calculated as:

    .. math:: - \\max_y \\sigma_y(f(x) / T)

    where :math:`\\sigma` is the softmax function, :math:`T` is the optimal temperature and :math:`\\sigma_y`
    indicates the :math:`y^{th}` value of the resulting probability vector.

    :see Paper: `ArXiv <https://arxiv.org/pdf/1706.04599.pdf>`__
    """
    def __init__(self, model: Module):
        """
        :param model: neural network to use
        """
        super(TemperatureScaling, self).__init__(model=model)
        self.t = torch.nn.Parameter(tensor(1.0))
        self._is_fitted = False

    def predict(self, x: Tensor) -> Tensor:
        return super().predict(x)

    def predict_features(self, logits: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RequiresFittingException()

        return super().predict_features(logits)

    def fit_features(self: Self, logits: Tensor, labels: Tensor) -> Self:
        """
        Optimize temperature using L-BFGS. Ignores OOD inputs.

        :param logits: logits
        :param labels: labels for logits
        """
        known = is_known(labels)

        if not known.any():
            raise ValueError(f"No IN samples")

        optimizer = LBFGS([self.t], lr=0.01, max_iter=50)

        device = self.t.device

        logits = logits[known].to(device)
        labels = labels[known].to(device)

        with torch.no_grad():
            loss = nll_loss(logits / self.t, labels).item()

        log.info(f"Initial T/NLL: {self.t.item():.3f}/{loss:.3f}")

        def closure():
            optimizer.zero_grad()
            loss = nll_loss(logits / self.t, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            loss = nll_loss(logits / self.t, labels).item()

        log.info(f"Optimal temperature: {self.t.item()}")
        log.info(f"NLL after scaling: {loss:.2f}'")

        self._is_fitted = True

        return self

    def fit(self: Self, loader: DataLoader, device: str = "cpu") -> Self:
        """
        Extracts features and optimizes the temperature.

        :param loader: data loader
        :param device: device used for extracting logits
        """
        z, y = extract_features(model=self.model, data_loader=loader, device=device)
        return self.fit_features(z, y)
