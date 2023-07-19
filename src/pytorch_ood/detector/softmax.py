"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.MaxSoftmax
    :members:

"""
from typing import Optional, TypeVar

import torch.nn
from torch import Tensor, tensor
from torch.nn.functional import nll_loss
from torch.nn import Module
from torch.optim import LBFGS
from torch.utils.data import DataLoader
import logging

from ..api import Detector, ModelNotSetException, RequiresFittingException
from pytorch_ood.utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class MaxSoftmax(Detector):
    """
    Implements the Maximum Softmax Probability (MSP) Thresholding baseline for OOD detection.

    Optionally, implements temperature scaling, which divides the logits by a constant temperature :math:`T`
    before calculating the softmax. The score is calculated as:

    .. math:: - \\max_y \\sigma_y(f(x) / T)

    where :math:`\\sigma` is the softmax function and :math:`\\sigma_y`  indicates the :math:`y^{th}` value of the
    resulting probability vector.

    :see Paper:
        `ArXiv <https://arxiv.org/abs/1610.02136>`_
    :see Implementation:
        `GitHub <https://github.com/hendrycks/error-detection>`_

    """

    def __init__(self, model: Module, t: Optional[float] = 1.0):
        """
        :param model: neural network to use
        :param t: temperature value :math:`T`. Default is 1.
        """
        super(MaxSoftmax, self).__init__()
        self.t = tensor(t)
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through model
        """
        if self.model is None:
            raise ModelNotSetException

        return self.predict_features(self.model(x))

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self

    def predict_features(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by the model
        """
        return MaxSoftmax.score(logits, self.t)

    @staticmethod
    def score(logits: Tensor, t: Optional[float] = 1.0) -> Tensor:
        """
        :param logits: logits for samples
        :param t: temperature value
        """
        return -logits.div(t).softmax(dim=1).max(dim=1).values

