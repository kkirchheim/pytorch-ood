import logging
from typing import Optional, TypeVar

import torch
from torch.utils.data import DataLoader

from ...api import Detector
from ...utils import TensorBuffer, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class OpenMax(Detector):
    """
    Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*.

    The methods determines a center :math:`\\mu_y` for each class in the logits space of a model, and then
    creates a statistical model of the distances of correct classified inputs.
    It uses extreme value theory to detect outliers by fitting a weibull function to the tail of the distance
    distribution.

    We use the activation of the *unknown* class as outlier score.

    .. warning:: This methods requires ``libmr`` to be installed, which is broken at the moment. You can only use it
       by installing ``cython`` and ``numpy``, and ``libmr`` manually afterwards.

    :see Paper: `ArXiv <https://arxiv.org/abs/1511.06233>`__
    :see Implementation: `GitHub <https://github.com/abhijitbendale/OSDN>`__
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tailsize: int = 25,
        alpha: int = 10,
        euclid_weight: float = 1.0,
    ):
        """
        :param model: neural network, assumed to output logits
        :param tailsize: length of the tail to fit the distribution to
        :param alpha: number of class activations to revise
        :param euclid_weight: weight for the euclidean distance.
        """
        self.model = model

        # we import it here because of its dependency to the broken libmr
        from .numpy import OpenMax as NPOpenMax

        self.openmax = NPOpenMax(tailsize=tailsize, alpha=alpha, euclid_weight=euclid_weight)

    def fit(self: Self, data_loader: DataLoader, device: Optional[str] = "cpu") -> Self:
        """
        Determines parameters of the weibull functions for each class.

        :param data_loader: Data to use for fitting
        :param device: Device used for calculations
        """
        z, y = OpenMax._extract(data_loader, self.model, device=device)
        return self.fit_features(z, y)

    def fit_features(self: Self, data_loader: DataLoader, device: Optional[str] = "cpu") -> Self:
        """
        Determines parameters of the weibull functions for each class.

        :param z: features
        :param y: class labels
        :param device: device to use
        :return:
        """
        z, y = z.cpu().numpy(), y.cpu().numpy()
        self.openmax.fit(z, y)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input, will be passed through the model to obtain logits
        """
        with torch.no_grad():
            z = self.model(x).cpu().numpy()

        return torch.tensor(self.openmax.predict(z)[:, 0])

