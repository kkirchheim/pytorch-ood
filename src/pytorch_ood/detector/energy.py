"""

..  autoclass:: pytorch_ood.detector.EnergyBased
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional

from torch import Tensor, logsumexp
from torch.nn import Module
from typing_extensions import Self

from ..api import DetectorInfo, LogitsDetector, Paper, Task


class EnergyBased(LogitsDetector):
    """
    Implements the Energy Score of *Energy-based Out-of-distribution Detection*.

    This method calculates the energy :math:`E(x)` for a vector of logits.
    The paper uses the negative energy as in-distribution score; the energy itself is used as outlier score,
    so larger values indicate OOD.

    .. math::
        E(x) = -T \\log{\\sum_i e^{f_i(x)/T}}

    where :math:`f_i(x)` indicates the :math:`i^{th}` logit value predicted by :math:`f`.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Energy-based Out-of-distribution Detection",
            venue="NeurIPS",
            year=2020,
            url="https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf",
            code="https://github.com/weitliu/energy_ood",
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
    )

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
        :param logits: logits given by the model, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :return: outlier scores of shape :math:`B` (or :math:`B \\times H \\times W`)
        """
        return EnergyBased.score(logits, t=self.t)

    @staticmethod
    def score(logits: Tensor, t: Optional[float] = 1.0) -> Tensor:
        """
        :param logits: logits of input, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :param t: temperature :math:`T`
        :return: energy, shape :math:`B` (or :math:`B \\times H \\times W`)
        """
        return -t * logsumexp(logits / t, dim=1)
