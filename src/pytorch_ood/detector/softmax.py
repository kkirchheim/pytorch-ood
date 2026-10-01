"""

..  autoclass:: pytorch_ood.detector.MaxSoftmax
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

import logging
from typing import Optional

from torch import Tensor, tensor
from torch.nn import Module
from typing_extensions import Self

from ..api import DetectorInfo, LogitsDetector, Paper, Task

log = logging.getLogger(__name__)


class MaxSoftmax(LogitsDetector):
    """
    Implements the Maximum Softmax Probability (MSP) Thresholding baseline for OOD detection.

    Optionally, implements temperature scaling, which divides the logits by a constant temperature :math:`T`
    before calculating the softmax. The score is calculated as:

    .. math:: - \\max_y \\sigma_y(f(x) / T)

    where :math:`\\sigma` is the softmax function and :math:`\\sigma_y`  indicates the :math:`y^{th}` value of the
    resulting probability vector, and :math:`f(x)` are the logits.
    The sign is flipped so that larger values indicate outliers.
    """

    info = DetectorInfo(
        paper=Paper(
            title="A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks",
            venue="ICLR",
            year=2017,
            url="https://arxiv.org/abs/1610.02136",
            code="https://github.com/hendrycks/error-detection",
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
    )

    def __init__(self, model: Optional[Module], t: float = 1.0):
        """
        :param model: neural network to use. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        :param t: temperature value :math:`T`
        """
        super(MaxSoftmax, self).__init__()
        self.t = tensor(t)
        self.model = model

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits of shape :math:`B \\times C` (or :math:`B \\times C \\times H \\times W`)
        :return: outlier scores of shape :math:`B` (or :math:`B \\times H \\times W`)
        """
        return MaxSoftmax.score(logits, self.t)

    @staticmethod
    def score(logits: Tensor, t: float = 1.0) -> Tensor:
        """
        Negative maximum softmax probability of the temperature-scaled logits.

        :param logits: logits of shape :math:`B \\times C` (or :math:`B \\times C \\times H \\times W`)
        :param t: temperature :math:`T`
        :return: negative maximum softmax probability, shape :math:`B` (or :math:`B \\times H \\times W`);
        """
        return -logits.div(t).softmax(dim=1).max(dim=1).values
