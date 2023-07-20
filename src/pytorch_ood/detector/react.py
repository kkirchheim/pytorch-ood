"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.ReAct
    :members:

"""
from typing import TypeVar, Callable
from torch import Tensor
import logging

from ..api import Detector
from pytorch_ood.detector import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class ReAct(Detector):
    """
    Implements ReAct from the paper
    *ReAct: Out-of-distribution Detection With Rectified Activations*.

    ReAct clips the activations in some layer of the network (backbone) and forward propagates the
    result through the remainder of the model (head).
    In the paper, ReAct is applied to the penultimate layer of the network.

    The output of the network is then passed to an outlier detector that maps the output of
    the model to outlier scores.

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = ReAct(
            backbone = model.features,
            head = model.fc,
            detector = EnergyBased.score
        )
        scores = detector(images)

    :see Paper: `ArXiv <https://arxiv.org/abs/2111.12797>`__
    """

    def __init__(self, backbone: Callable[[Tensor], Tensor], head: Callable[[Tensor], Tensor],
                 threshold: float = 1.0, detector: Callable[[Tensor], Tensor] = None):
        """
        :param backbone: first part of model to use, should output feature maps
        :param head: second part of model used after applying ash, should output logits
        :param threshold: cutoff for activations
        :param detector: detector that maps outputs to outlier scores. Default is energy based.
        """
        self.backbone = backbone
        self.head = head
        self.threshold = threshold
        self.detector = detector or EnergyBased.score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through network
        """
        x = self.backbone(x)
        x = x.clip(max=self.threshold)
        x = self.head(x)
        return self.detector(x)

    def predict_features(self, x: Tensor) -> Tensor:
        """
        :raises: NotImplementedError
        """
        raise NotImplementedError

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        raise self

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self
