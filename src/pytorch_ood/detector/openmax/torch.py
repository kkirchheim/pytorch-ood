"""
Torch wrapper for a numpy implementation of openmax.
"""

import logging
from typing import Optional, TypeVar

import torch
from torch import Tensor
from torch.nn import Module

from ...api import LogitsDetector, ModelNotSetException
from .numpy import OpenMax as NumpyOpenMax

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class OpenMax(LogitsDetector):
    """
    Implementation of the OpenMax Layer as proposed in the paper *Towards Open Set Deep Networks*.

    The method determines a center :math:`\\mu_y` for each class in the logits space of a model, and then
    creates a statistical model of the distances of correct classified inputs.
    It uses extreme value theory to detect outliers by fitting a weibull function to the tail of the distance
    distribution.

    We use the pseudo-activation of the *unknown* class as outlier score.

    :see Paper: `ArXiv <https://arxiv.org/abs/1511.06233>`__
    :see Implementation: `GitHub <https://github.com/abhijitbendale/OSDN>`__
    """

    requires_fit = True

    #: grid explored by :class:`pytorch_ood.utils.GridSearch`. ``alpha``/``euclid_weight``
    #: anchored around the fixed values used by the OpenOOD reference implementation
    #: (weibull_alpha=3, eu_weight=0.5); ``tailsize`` around the paper/OpenOOD default of
    #: 20-25, since OpenOOD itself does not sweep any of these three.
    hyperparameter_space = {
        "tailsize": [10, 20, 30, 40, 50],
        "alpha": [1, 3, 5, 10, 15, 20],
        "euclid_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    def __init__(
        self,
        model: Optional[Module],
        tailsize: int = 25,
        alpha: int = 10,
        euclid_weight: float = 1.0,
    ):
        """
        :param model: neural network, assumed to output logits. Can be ``None`` when using
            ``fit_logits(...)`` and ``predict_logits(...)`` directly.
        :param tailsize: length of the tail to fit the distribution to
        :param alpha: number of class activations to revise
        :param euclid_weight: weight for the Euclidean distance.
        """
        self.model = model
        self.tailsize = tailsize
        self.alpha = alpha
        self.euclid_weight = euclid_weight

    def fit_logits(self: Self, logits: Tensor, y: Tensor) -> Self:
        """
        Determines parameters of the weibull functions for each class.

        :param logits: logits given by the model
        :param y: class labels
        :return:
        """
        # Built here (not in __init__) so that GridSearch.set_hyperparameters -- which
        # only assigns self.tailsize/alpha/euclid_weight, per hyperparameter_space --
        # takes effect: every GridSearch candidate re-fits after setting hyperparameters,
        # so a fresh, correctly-configured NumpyOpenMax is guaranteed at fit time.
        self._openmax = NumpyOpenMax(
            tailsize=self.tailsize, alpha=self.alpha, euclid_weight=self.euclid_weight
        )
        logits, y = logits.cpu().numpy(), y.cpu().numpy()
        self._openmax.fit(logits, y)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through the model to get logits
        """
        if self.model is None:
            raise ModelNotSetException

        device = self.device
        if device is not None:
            x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)

        return self.predict_logits(logits)

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by model
        """
        device = self.device or logits.device
        logits = logits.detach().cpu().numpy()
        return torch.tensor(self._openmax.predict(logits)[:, 0], device=device)
