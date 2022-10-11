"""

..  autoclass:: pytorch_ood.detector.MaxSoftmax
    :members:

"""
from typing import Optional

import torch

from ..api import Detector


class MaxSoftmax(Detector):
    """
    Implements the Maximum Softmax Thresholding Baseline for OOD detection.

    Optionally, implements temperature scaling, which divides the logits by a constant temperature :math:`T`
    before calculating the softmax.

    .. math:: - \\max_y \\sigma_y(f(x) / T)

    where :math:`\\sigma` is the softmax function and :math:`\\sigma_y`  indicates the :math:`y^{th}` value of the
    resulting probability vector.

    :see Paper:
        `ArXiv <https://arxiv.org/abs/1610.02136>`_
    :see Implementation:
        `GitHub <https://github.com/hendrycks/error-detection>`_

    """

    def __init__(self, model: torch.nn.Module, t: Optional[float] = 1):
        """
        :param model: neural network to use
        :param t: temperature value T. Default is 1.
        """
        super(MaxSoftmax, self).__init__()
        self.t = t
        self.model = model

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: model input, will be passed through neural network
        """
        return self.score(self.model(x), t=1)

    def fit(self, *args, **kwargs):
        """
        Not required

        """
        pass

    @staticmethod
    def score(logits: torch.Tensor, t: Optional[float] = 1) -> torch.Tensor:
        """
        :param logits: logits for samples
        :param t: temperature value
        """
        return -logits.div(t).softmax(dim=1).max(dim=1).values
