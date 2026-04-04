"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightred?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Entropy
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional, TypeVar

from torch import Tensor
from torch.nn import Module

from ..api import LogitsDetector

Self = TypeVar("Self")


class Entropy(LogitsDetector):
    """
    Implements Entropy-based OOD detection.

    This methods calculates the entropy based on the logits of a classifier.
    Higher entropy means more uniformly distributed posteriors, indicating larger uncertainty.
    Entropy is calculated as

    .. math::
        H(x) = - \\sum_i^C  \\sigma_i(f(x)) \\log( \\sigma_i(f(x)) )

    where :math:`\\sigma_i` indicates the :math:`i^{th}` softmax value and :math:`C` is the number of classes.

    """

    def __init__(self, model: Optional[Module]):
        """
        :param model: the model :math:`f`. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        """
        super(Entropy, self).__init__()
        self.model = model

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by your model
        """
        return self.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits of input
        """
        p = logits.softmax(dim=1).clip(1e-7, 1)
        return -(p.log() * p).sum(dim=1)
