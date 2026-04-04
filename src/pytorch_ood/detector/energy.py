"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.EnergyBased
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional, TypeVar

from torch import Tensor, logsumexp
from torch.nn import Module

from ..api import LogitsDetector

Self = TypeVar("Self")


class EnergyBased(LogitsDetector):
    """
    Implements the Energy Score of  *Energy-based Out-of-distribution Detection*.

    This methods calculates the negative energy for a vector of logits.
    This value can be used as outlier score.

    .. math::
        E(x) = -T \\log{\\sum_i e^{f_i(x)/T}}

    where :math:`f_i(x)` indicates the :math:`i^{th}` logit value predicted by :math:`f`.

    :see Paper:
        `NeurIPS <https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf>`__

    :see Implementation:
        `GitHub <https://github.com/wetliu/energy_ood>`__

    """

    def __init__(self, model: Optional[Module], t: Optional[float] = 1.0):
        """
        :param model: neural network to use. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        :param t: Temperature value :math:`T`. Default is 1.
        """
        super(EnergyBased, self).__init__()
        self.t: float = t  #: Temperature
        self.model = model

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by the model
        """
        return EnergyBased.score(logits, t=self.t)

    @staticmethod
    def score(logits: Tensor, t: Optional[float] = 1.0) -> Tensor:
        """
        :param logits: logits of input
        :param t: temperature value
        """
        return -t * logsumexp(logits / t, dim=1)
