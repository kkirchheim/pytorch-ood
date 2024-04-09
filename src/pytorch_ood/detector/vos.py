"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.VOSBased
    :members:
    :exclude-members: fit, fit_features
"""
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")


class VOSBased(Detector):
    # TODO ggf Link zu VOS Loss
    """
    Implements the VOS-Energy Score of  *VOS: LEARNING WHAT YOU DON’T KNOW BY
    VIRTUAL OUTLIER SYNTHESIS*.

    This methods calculates the energy for a vector of logits and the energy weights (you only get from the training with VOS_Loss).
    This value can be used as outlier score.

    .. math::
        E(x) = - \\log{\\sum_i w_{i} e^{f_i(x)}}

    where :math:`f_i(x)` indicates the :math:`i^{th}` logit value predicted by :math:`f` and :math `w` indicates the weights energy.

    :see Paper:
        ` <https://openreview.net/pdf?id=TW7d65uYu5M>`__

    :see Implementation:
        `GitHub <https://github.com/deeplearning-wisc/vos/>`__

    """

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def __init__(self, model: torch.nn.Module, weights_energy: torch.nn.Module):
        """
        :param t: Temperature value :math:`T`. Default is 1.
        """
        super(VOSBased, self).__init__()

        self.model = model
        self.weights_energy = weights_energy

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative energy for inputs

        :param x: input tensor, will be passed through model

        :return: Energy score
        """
        if self.model is None:
            raise ModelNotSetException

        return self.score(self.model(x), self.weights_energy)

    def predict_features(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by your model
        """
        return self.score(logits)

    @staticmethod
    def score(logits: torch.Tensor, weights_energy: torch.nn.Module) -> torch.Tensor:
        """
        :param logits: logits of input
        :param weights_energy: energy weights as torch.nn.module
        """
        # Permutation depends on shape of logits
        tmp_scores_ = logits.permute(0, 2, 3, 1)

        conf = torch.log(
            torch.sum(
                (F.relu(weights_energy.weight) * torch.exp(tmp_scores_)), dim=3, keepdim=False
            )
        )

        return -conf
