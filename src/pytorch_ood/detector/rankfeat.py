"""

..  autoclass:: pytorch_ood.detector.RankFeat
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit
"""

import logging
from typing import Callable, Optional

import torch
from torch import Tensor
from typing_extensions import Self

from ..api import DetectorInfo, FeatureMapsDetector, Paper, Task
from .energy import EnergyBased

log = logging.getLogger(__name__)


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
    Implements RankFeat from *RankFeat: Rank-1 Feature Removal for Out-of-Distribution Detection*.

    RankFeat removes the dominant rank-1 component from intermediate feature maps
    via SVD before forwarding through the remainder of the network. The intuition is
    that the leading singular vector captures generic, class-agnostic patterns shared
    between ID and OOD data. Removing it exposes subtler, class-specific structure
    that the energy score can exploit for better discrimination.

    Concretely, given a feature map :math:`\\mathbf{X} \\in \\mathbb{R}^{C \\times HW}`,
    the method computes its (economy) SVD and subtracts the rank-1 approximation, where
    :math:`\\sigma_1, \\mathbf{u}_1, \\mathbf{v}_1` are the largest singular value and the corresponding left and
    right singular vectors:

    .. math::
        \\mathbf{X}' = \\mathbf{X} - \\sigma_1 \\, \\mathbf{u}_1 \\, \\mathbf{v}_1^\\top

    The modified features :math:`\\mathbf{X}'` are then forwarded through the classification
    head, and the resulting logits are mapped to outlier scores by ``detector``, by default
    :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.

    Like :class:`~pytorch_ood.detector.ASH` and :class:`~pytorch_ood.detector.ReAct`,
    the model must be split into a ``backbone`` (up to and including the target
    convolutional block) and a ``head`` (the remaining layers including the classifier).

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = RankFeat(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )
        scores = detector(images)
    """

    info = DetectorInfo(
        paper=Paper(
            title="RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection",
            venue="NeurIPS",
            year=2022,
            url="https://arxiv.org/abs/2209.08590",
            code="https://github.com/KingJamesSong/RankFeat",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        detector: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        :param backbone: first part of the model, should output 4-D feature maps of shape :math:`B \\times C \\times H \\times W`
        :param head: second part of the model applied after rank-1 removal, should output logits
        :param detector: scoring function mapping logits of shape :math:`B \\times C` to outlier scores
            of shape :math:`B`. Defaults to :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.
        """
        self.backbone = backbone
        self.head = head
        self.detector = detector or EnergyBased.score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input batch, will be passed through the backbone and head. Must yield 4-D feature maps
            of shape :math:`B \\times C \\times H \\times W`.
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
        Removes the rank-1 component of each sample's feature map and scores the resulting logits.

        :param feature_maps: feature maps of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B`
        """
        x = _remove_rank1(feature_maps)
        x = self.head(x)
        return self.detector(x)
