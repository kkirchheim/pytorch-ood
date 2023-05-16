"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightred?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Entropy
    :members:

"""
from typing import Optional, TypeVar

from torch import Tensor
from torch.nn import Module

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")


class Entropy(Detector):
    """
    Implements Entropy-based OOD detection.

    This methods calculates the entropy based on the logits of a classifier.
    Higher entropy means more uniformly distributed posteriors, indicating larger uncertainty.
    Entropy is calculated as

    .. math::
        H(x) = - \\sum_i^C  \\sigma_i(f(x)) \\log( \\sigma_i(f(x)) )

    where :math:`\\sigma_i` indicates the :math:`i^{th}` softmax value and :math:`C` is the number of classes.

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

    def __init__(self, model: Module):
        """
        :param model: the model :math:`f`
        """
        super(Entropy, self).__init__()
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate entropy for inputs

        :param x: input tensor, will be passed through model

        :return: Entropy score
        """
        if self.model is None:
            raise ModelNotSetException

        return self.score(self.model(x))

    def predict_features(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by your model
        """
        return self.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits of input
        """
        p = logits.softmax(dim=1)
        return -(p.log() * p).sum(dim=1)
