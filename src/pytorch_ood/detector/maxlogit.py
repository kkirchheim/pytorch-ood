"""
.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.MaxLogit
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


class MaxLogit(LogitsDetector):
    """
    Implements the Max Logit Method for OOD Detection as proposed in
    *Scaling Out-of-Distribution Detection for Real-World Settings*.

    .. math:: - \\max_y f_y(x)

    where :math:`f_y(x)` indicates the :math:`y^{th}` logits value predicted by :math:`f`.

    :see Paper:
       `ArXiv <https://arxiv.org/abs/1911.11132>`__
    """

    def __init__(self, model: Optional[Module]):
        """
        :param model: neural network to use. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        """
        super(MaxLogit, self).__init__()
        self.model = model

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits as given by the model
        """
        return MaxLogit.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits for samples
        """
        return -logits.max(dim=1).values
