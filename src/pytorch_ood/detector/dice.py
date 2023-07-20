"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.DICE
    :members:

"""
from typing import TypeVar, Callable

import torch.nn
from torch import Tensor
from torch.utils.data import DataLoader
import logging
import numpy as np

from ..api import Detector
from pytorch_ood.utils import extract_features, is_known
from pytorch_ood.detector import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class DICE(Detector):
    """
    Implements DICE from the paper
    *DICE: Leveraging Sparsification for Out-of-Distribution Detection*.

    :see Paper: `ArXiv <https://arxiv.org/abs/2111.09805>`__
    """

    def __init__(self, model: Callable[[Tensor], Tensor], w: torch.Tensor, b: torch.Tensor, p: float,
                 detector: Callable[[Tensor], Tensor] = None):
        """
        :param model: feature extractor
        :param w: weights of last layer
        :param b: bias of last layer
        :param p: percentile of weights to drop
        """
        self.model = model
        self.weight = w.detach().cpu()
        self.bias = b.detach().cpu()
        self.percentile = p
        self.detector = detector or EnergyBased.score

        self._is_fitted = False

        self.masked_w = None
        self.threshold = None
        self.mean_activation = None

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through network
        """
        z = self.model(x)
        return self.predict_features(z)

    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features
        """
        vote = x[:, None, :] * self.masked_w.to(x.device)
        output = vote.sum(2) + self.bias.to(x.device)
        score = self.detector(output)
        return score

    def fit_features(self: Self, z: Tensor, y: Tensor) -> Self:
        """
        Calculates the masked weights. OOD Inputs will be ignored.

        :param z: features
        :param y: labels.
        """
        known = is_known(y)

        if not known.any():
            raise ValueError(f"No IN data")

        z = z[known]

        self.mean_activation = z.mean(dim=0)

        contrib = self.mean_activation[None, :] * self.weight
        self.threshold = np.percentile(contrib, self.percentile)
        log.info(f"Threshold is {self.threshold:.2f}")
        self.masked_w = torch.where(contrib > self.threshold, self.weight, 0).to(z.device)
        self._is_fitted = True
        return self

    def fit(self: Self, loader: DataLoader, device: str = "cpu") -> Self:
        """
        :param loader: data loader to extract features from. OOD inputs will be ignored.
        :param device: device to use for feature extraction
        """
        z, y = extract_features(loader, self.model, device=device)
        self.fit_features(z, y)
        return self
