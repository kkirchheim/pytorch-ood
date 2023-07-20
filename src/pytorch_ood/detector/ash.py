"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.ASH
    :members:

"""
from typing import TypeVar, Callable

import torch.nn
from torch import Tensor

import logging
import numpy as np

from ..api import Detector
from pytorch_ood.detector import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


def ash_b(x: Tensor, percentile: float = 0.65) -> Tensor:
    assert x.dim() == 4
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x: Tensor, percentile: float = 0.65) -> Tensor:
    assert x.dim() == 4

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x: Tensor, percentile: float = 0.65) -> Tensor:
    assert x.dim() == 4
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


class ASH(Detector):
    """
    Implements ASH from the paper
    *Extremely Simple Activation Shaping for Out-of-Distribution Detection*.

    ASH prunes the activations in some layer of the network (backbone) by removing a certain percentile of
    the highest activations. The remaining activations are modified, depending on the particular variant selected, and
    propagated through the remainder (head) of the network.
    Then uses the energy based outlier score.
    This approach has been shown to increase OOD detection rates while maintaining IN accuracy.

    * ASH-P: only prune, do not modify
    * ASH-B: binarize remaining activations
    * ASH-S: rescale remaining activations

    The paper applies ASH after the last average pooling layer.

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = ASH(
            backbone = model.features_before_pool,
            head = model.forward_from_before_pool,
            detector=EnergyBased.score
        )
        scores = detector(images)

    :see Paper: `ICLR 2023 <https://openreview.net/pdf?id=ndYXTEL6cZz>`__
    :see Website: `github.io <https://andrijazz.github.io/ash/>`__
    """

    variants = {
        "ash-s": ash_s,
        "ash-p": ash_p,
        "ash-b": ash_b,
    }

    def __init__(self, backbone: Callable[[Tensor], Tensor], head: Callable[[Tensor], Tensor],
                 variant="ash-s", percentile: float = 0.65, detector: Callable[[Tensor], Tensor] = None):
        """
        :param variant: one of ``ash-p``, ``ash-b``, ``ash-s``
        :param backbone: first part of model to use, should output feature maps
        :param head: second part of model used after applying ash, should output logits
        :param percentile: amount of activations to modify
        :param detector: detector that maps model outputs to outlier scores. Default is Energy based.
        """
        assert variant in self.variants

        self.backbone = backbone
        self.head = head
        self.percentile = percentile
        self.ash: Callable[[Tensor, float], Tensor] = self.variants[variant]
        self.detector = detector or EnergyBased.score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through network
        """
        x = self.backbone(x)
        x = self.ash(x, self.percentile)
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
