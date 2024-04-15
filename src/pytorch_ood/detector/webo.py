"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.WeightedEBO
    :members:
    :exclude-members: fit, fit_features
"""
from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")


class WeightedEBO(Detector):
    """
    Implements the Weighted Energy Based Score of  *VOS: Learning what you don’t know by virtual outlier synthesis*.

    This methods calculates the energy for a vector of logits and the weights (you only get from the training with :class:`pytorch_ood.loss.VOSRegLoss`).
    This value can be used as outlier score.

    .. math::
        E(x) = - \\log{\\sum_i w_{i} e^{f_i(x)}}

    where :math:`f_i(x)` indicates the :math:`i^{th}` logit value predicted by :math:`f` and :math:`w` indicates the weights.

    Example Code:

    .. code :: python

        weights = torch.nn.Linear(num_classes, 1))
        detector = WeightedEBO(model, weights)
        scores = detector(images)


    :see Paper:
        `ArXiv <https://arxiv.org/pdf/2202.01197.pdf>`__

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

    def __init__(self, model: torch.nn.Module, weights: torch.nn.Linear):
        """
        :param model: neural network to use, is assumed to output features
        :param weights: neural network layer, with num_classes inputs.
        """
        super(WeightedEBO, self).__init__()

        self.model = model
        self.weights = weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted energy for inputs

        :param x: input tensor, will be passed through model

        :return: Weighted Energy score
        """
        if self.model is None:
            raise ModelNotSetException

        return self.score(self.model(x), self.weights)

    def predict_features(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by your model
        """
        return self.score(logits, self.weights)

    @staticmethod
    def score(logits: torch.Tensor, weights: torch.nn.Module) -> torch.Tensor:
        """
        :param logits: logits of input
        :param weights: weights as torch.nn.module
        """

        if len(logits.shape) == 2:
            conf = torch.log(
                torch.sum((F.relu(weights.weight) * torch.exp(logits)), dim=1, keepdim=False)
            )

            return -conf
        elif len(logits.shape) == 4:
            # Permutation depends on shape of logits

            tmp_scores_ = logits.permute(0, 2, 3, 1)

            conf = torch.log(
                torch.sum(
                    (F.relu(weights.weight) * torch.exp(tmp_scores_)),
                    dim=3,
                    keepdim=False,
                )
            )

            return -conf
        else:
            raise ValueError(f"Unsupported input shape: {logits.shape}")
