"""

..  autoclass:: pytorch_ood.detector.MaxLogit
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional

from torch import Tensor
from torch.nn import Module
from typing_extensions import Self

from ..api import DetectorInfo, LogitsDetector, Paper, Task


class MaxLogit(LogitsDetector):
    """
    Implements the Max Logit Method for OOD Detection as proposed in
    *Scaling Out-of-Distribution Detection for Real-World Settings*.

    .. math:: - \\max_y f_y(x)

    where :math:`f_y(x)` indicates the :math:`y^{th}` logits value predicted by :math:`f`.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Scaling Out-of-Distribution Detection for Real-World Settings",
            venue="ICML",
            year=2022,
            url="https://arxiv.org/abs/1911.11132",
            code=None,
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
    )

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
