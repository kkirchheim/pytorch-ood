"""

..  autoclass:: pytorch_ood.detector.ASH
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""

import logging
from typing import Callable

import numpy as np
import torch.nn
from torch import Tensor
from typing_extensions import Self

from ..api import DetectorInfo, FeatureMapsDetector, Paper, Task
from ..utils.utils import _check_fraction
from .energy import EnergyBased

log = logging.getLogger(__name__)


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


class ASH(FeatureMapsDetector):
    """
    Implements ASH from the paper
    *Extremely Simple Activation Shaping for Out-of-Distribution Detection*.

    ASH prunes the activations in some layer of the network (backbone output) by setting the lowest fraction
    ``percentile`` of the activations of each sample to zero. The remaining (highest) activations are modified,
    depending on the particular variant selected, and propagated through the remainder (head) of the network.
    The resulting logits are then mapped to outlier scores by ``detector``, which defaults to
    :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.
    This approach has been shown to increase OOD detection rates while maintaining ID accuracy.

    * ASH-P: only prune, do not modify
    * ASH-B: binarize remaining activations
    * ASH-S: rescale remaining activations

    ASH is applied to a single layer of the network. In the example below, the feature maps before the final
    average pooling are shaped.

    .. note::
        The activations passed to the detector are modified in place, so cached feature maps should not be
        reused afterwards. Feature maps have to be 4-dimensional; pooled features of shape
        :math:`B \\times C` can be passed as :math:`B \\times C \\times 1 \\times 1`.

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = ASH(
            backbone = model.feature_maps,
            head = model.forward_feature_maps,
            detector=EnergyBased.score
        )
        scores = detector(images)

    :see Website: `Project page <https://andrijazz.github.io/ash/>`__
    """

    info = DetectorInfo(
        paper=Paper(
            title="Extremely Simple Activation Shaping for Out-of-Distribution Detection",
            venue="ICLR",
            year=2023,
            url="https://openreview.net/pdf?id=ndYXTEL6cZz",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    variants = {
        "ash-s": ash_s,
        "ash-p": ash_p,
        "ash-b": ash_b,
    }

    # matches the percentile sweep used by OpenOOD (expressed here as fractions in [0, 1])
    #: Default search space for :class:`pytorch_ood.utils.GridSearch` (fractions in :math:`[0, 1]`).
    hyperparameter_space = {
        "percentile": [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    }

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        variant="ash-s",
        percentile: float = 0.65,
        detector: Callable[[Tensor], Tensor] = None,
    ):
        """
        :param variant: one of ``ash-p``, ``ash-b``, ``ash-s``
        :param backbone: first part of model to use, should output feature maps of shape
            :math:`B \\times C \\times H \\times W`
        :param head: second part of model used after applying ASH, should output logits of shape
            :math:`B \\times C`
        :param percentile: fraction in :math:`[0, 1]` of the activations (per sample) that is set to zero,
            starting from the smallest. Default is 0.65.
        :param detector: maps the logits returned by ``head`` to outlier scores. Default is
            :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.
        """
        assert variant in self.variants

        self.backbone = backbone
        self.head = head
        self.percentile = _check_fraction("percentile", percentile)
        self.ash: Callable[[Tensor, float], Tensor] = self.variants[variant]
        self.detector = detector or EnergyBased.score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through ``backbone``
        :return: outlier scores of shape :math:`B`
        """
        device = self.device
        if device is not None:
            x = x.to(device)
        x = self.backbone(x)
        return self.predict_feature_maps(x)

    @torch.no_grad()
    def predict_feature_maps(self, feature_maps: Tensor) -> Tensor:
        """
        :param feature_maps: activations of the backbone, shape
            :math:`B \\times C \\times H \\times W`. Modified in place.
        :return: outlier scores of shape :math:`B`
        """
        x = self.ash(feature_maps, self.percentile)
        x = self.head(x)
        return self.detector(x)
