"""

..  autoclass:: pytorch_ood.detector.Entropy
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


class Entropy(LogitsDetector):
    """
    Implements Entropy-based OOD detection.

    This method calculates the entropy based on the logits of a classifier.
    Higher entropy means more uniformly distributed posteriors, indicating larger uncertainty.
    Entropy is calculated as

    .. math::
        H(x) = - \\sum_i^C  \\sigma_i(f(x)) \\log( \\sigma_i(f(x)) )

    where :math:`\\sigma_i` indicates the :math:`i^{th}` softmax value and :math:`C` is the number of classes.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Entropy Maximization and Meta Classification for Out-of-Distribution Detection in Semantic Segmentation",
            venue="ICCV",
            year=2021,
            url="https://arxiv.org/abs/2012.06575",
            code=None,
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
    )

    def __init__(self, model: Optional[Module]):
        """
        :param model: the model :math:`f`. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        """
        super(Entropy, self).__init__()
        self.model = model

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by your model, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :return: entropy of the softmax distribution, shape :math:`B` (or :math:`B \\times H \\times W`)
        """
        return self.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits of input, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :return: entropy of the softmax distribution, shape :math:`B` (or :math:`B \\times H \\times W`).
            Probabilities are clipped to :math:`[10^{-7}, 1]` before taking the
            logarithm for numerical stability.
        """
        p = logits.softmax(dim=1).clip(1e-7, 1)
        return -(p.log() * p).sum(dim=1)
