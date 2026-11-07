"""Builds every detector that claims segmentation, for (B, 3, H, W) inputs."""

import torch
from torch import nn

from pytorch_ood import detector
from tests.helpers.losses import C
from tests.helpers.model import SegmentationModel


class _ConvBackbone(nn.Module):
    """Feature maps (B, 8, H, W) and a per-pixel head (B, C, H, W), for segmentation."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU())
        self.head = nn.Conv2d(8, C, 1)


def _with_backbone(cls):
    def build():
        model = _ConvBackbone().eval()
        return cls(backbone=model.features, head=model.head)

    return build


# Builds each detector that claims segmentation, for (B, 3, H, W) inputs.
SEGMENTATION_DETECTORS = {
    detector.EnergyBased: lambda: detector.EnergyBased(SegmentationModel().eval()),
    detector.Entropy: lambda: detector.Entropy(SegmentationModel().eval()),
    detector.GEN: lambda: detector.GEN(SegmentationModel().eval()),
    detector.MaxLogit: lambda: detector.MaxLogit(SegmentationModel().eval()),
    detector.MaxSoftmax: lambda: detector.MaxSoftmax(SegmentationModel().eval()),
    detector.MCD: lambda: detector.MCD(SegmentationModel().eval(), samples=4),
    detector.ReAct: _with_backbone(detector.ReAct),
    detector.WeightedEBO: lambda: detector.WeightedEBO(SegmentationModel().eval(), torch.ones(C)),
}
