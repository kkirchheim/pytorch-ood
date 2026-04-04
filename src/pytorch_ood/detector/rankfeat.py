"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.RankFeat
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""

import logging
from typing import Callable, TypeVar

import torch
from torch import Tensor

from ..api import FeatureMapsDetector
from .energy import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


def _remove_rank1(x: Tensor) -> Tensor:
    """
    Remove the rank-1 approximation from a batch of feature maps.

    :param x: feature maps of shape ``(B, C, H, W)``
    :return: feature maps with rank-1 component subtracted, same shape
    """
    B, C, H, W = x.shape
    # Reshape to (B, C, H*W) — each sample is a (C, H*W) matrix
    m = x.view(B, C, H * W)
    # Economy SVD: only need the first singular triplet
    u, s, v = torch.linalg.svd(m, full_matrices=False)
    # Subtract rank-1 approximation:  s_1 * u_1 @ v_1^T
    rank1 = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
    m = m - rank1
    return m.view(B, C, H, W)


class RankFeat(FeatureMapsDetector):
    """
    Implements RankFeat from *Rankfeat: Rank-1 Feature Removal for Out-of-Distribution Detection*.

    RankFeat removes the dominant rank-1 component from intermediate feature maps
    via SVD before forwarding through the remainder of the network. The intuition is
    that the leading singular vector captures generic, class-agnostic patterns shared
    between ID and OOD data. Removing it exposes subtler, class-specific structure
    that the energy score can exploit for better discrimination.

    Concretely, given a feature map :math:`\\mathbf{X} \\in \\mathbb{R}^{C \\times HW}`,
    the method computes its (economy) SVD and subtracts the rank-1 approximation:

    .. math::
        \\mathbf{X}' = \\mathbf{X} - \\sigma_1 \\, \\mathbf{u}_1 \\, \\mathbf{v}_1^\\top

    The modified features :math:`\\mathbf{X}'` are then forwarded through the classification
    head, and the resulting logits are scored with the energy function.

    Like :class:`~pytorch_ood.detector.ASH` and :class:`~pytorch_ood.detector.ReAct`,
    the model must be split into a ``backbone`` (up to and including the target
    convolutional block) and a ``head`` (the remaining layers including the classifier).

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = RankFeat(
            backbone=model.features_before_pool,
            head=model.forward_from_before_pool,
        )
        scores = detector(images)

    :see Paper: `NeurIPS 2022 <https://arxiv.org/abs/2209.08590>`__
    :see Implementation: `GitHub <https://github.com/KingJamesSong/RankFeat>`__
    """

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        detector: Callable[[Tensor], Tensor] = None,
    ):
        """
        :param backbone: first part of the model, should output 4-D feature maps ``(B, C, H, W)``
        :param head: second part of the model applied after rank-1 removal, should output logits
        :param detector: scoring function mapping logits to outlier scores.
            Default is :func:`~pytorch_ood.detector.EnergyBased.score`.
        """
        self.backbone = backbone
        self.head = head
        self.detector = detector or EnergyBased.score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through network
        :return: outlier scores
        """
        x = self.backbone(x)
        return self.predict_feature_maps(x)

    def predict_feature_maps(self, x: Tensor) -> Tensor:
        x = _remove_rank1(x)
        x = self.head(x)
        return -self.detector(x)
