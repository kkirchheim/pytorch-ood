"""

.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge
.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.SCALE
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""

import logging
from typing import Callable, TypeVar

import numpy as np
import torch.nn
from torch import Tensor

from ..api import FeatureMapsDetector
from .energy import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


def scale(x: Tensor, percentile: float = 0.65) -> Tensor:
    """
    Computes the SCALE per-sample scaling factor and applies it to the
    (unpruned) activations.

    The scaling factor is :math:`\\exp(s_1 / s_2)`, where :math:`s_1` is the sum
    over all activations and :math:`s_2` is the sum over the top-``(1 - percentile)``
    activations. Unlike ASH-S, the activations are **not** pruned: the top-k
    selection is used only to derive the scaling factor, which is then applied
    to the original, full activation tensor.

    :param x: feature maps of shape :math:`(B, C, H, W)`
    :param percentile: fraction of activations to consider as pruned when
        computing the scaling factor (expressed in ``[0, 1]``)
    """
    assert x.dim() == 4
    assert 0.0 <= percentile <= 1.0

    b, c, h, w = x.shape

    # sum over all activations per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)

    # sum over the surviving (top-k) activations per sample
    s2 = v.sum(dim=1)

    scale = s1 / s2
    return x * torch.exp(scale[:, None, None, None])


class SCALE(FeatureMapsDetector):
    """
    Implements SCALE from the paper
    *Scaling for Training Time and Post-hoc Out-of-distribution Detection Enhancement*.

    SCALE scales the activations in some layer of the network (backbone) by a
    per-sample factor and propagates the result through the remainder (head) of
    the network. Then uses the energy based outlier score.

    The scaling factor :math:`\\exp(s_1 / s_2)` is derived from the ratio between the
    sum over all activations (:math:`s_1`) and the sum over the largest
    ``(1 - percentile)`` fraction of activations (:math:`s_2`). In contrast to
    :class:`ASH <pytorch_ood.detector.ASH>` (ASH-S), the activations themselves are
    **not pruned**: the paper finds that activation pruning is detrimental to
    OOD detection, while scaling the *full* activations enhances it.

    The paper applies SCALE after the last average pooling layer.

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = SCALE(
            backbone = model.feature_maps,
            head = model.forward_feature_maps,
            detector = EnergyBased.score
        )
        scores = detector(images)

    :see Paper: `ICLR 2024 <https://arxiv.org/abs/2310.00227>`__
    :see Implementation: `GitHub <https://github.com/kai422/SCALE>`__
    """

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`, matching the
    #: percentile sweep used by OpenOOD (expressed here as fractions in ``[0, 1]``).
    hyperparameter_space = {
        "percentile": [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    }

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        percentile: float = 0.65,
        detector: Callable[[Tensor], Tensor] = None,
    ):
        """
        :param backbone: first part of model to use, should output feature maps
        :param head: second part of model used after applying scaling, should output logits
        :param percentile: fraction of activations treated as pruned when computing the scaling factor
        :param detector: detector that maps model outputs to outlier scores. Default is Energy based.
        """
        self.backbone = backbone
        self.head = head
        self.percentile = percentile
        self.detector = detector or EnergyBased.score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through network
        """
        device = self.device
        if device is not None:
            x = x.to(device)
        x = self.backbone(x)
        return self.predict_feature_maps(x)

    @torch.no_grad()
    def predict_feature_maps(self, x: Tensor) -> Tensor:
        x = scale(x, self.percentile)
        x = self.head(x)
        return self.detector(x)
