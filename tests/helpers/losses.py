"""Builds every loss with matching inputs, for classification and segmentation shapes."""

import torch
from torch import nn

from pytorch_ood import loss
from pytorch_ood.api import Task

B, C, D = 8, 3, 5  # batch size, classes, feature dimension
SHAPES = {Task.CLASSIFICATION: (B,), Task.SEGMENTATION: (2, 4, 4)}


def _outputs(shape, channels):
    # channels second, as in (B, C) and (B, C, H, W)
    return torch.randn(shape[0], channels, *shape[1:])


def _targets(shape, outliers=True):
    targets = torch.arange(torch.Size(shape).numel()).reshape(shape) % (C + 1) - 1
    return targets if outliers else targets.clamp(min=0)


# Builds a loss and its arguments for outputs of the given shape (see SHAPES).
LOSS_INPUTS = {
    loss.BackgroundClassLoss: lambda s: (
        loss.BackgroundClassLoss(n_classes=C),
        (_outputs(s, C + 1), _targets(s)),
    ),
    loss.CACLoss: lambda s: (
        loss.CACLoss(n_classes=C),
        (_outputs(s, C).abs(), _targets(s, False)),
    ),
    loss.CenterLoss: lambda s: (
        loss.CenterLoss(n_classes=C, n_dim=D),
        (_outputs(s, C).abs(), _targets(s, False)),
    ),
    loss.ConfidenceLoss: lambda s: (
        loss.ConfidenceLoss(),
        (_outputs(s, C), torch.rand(s[0], 1), _targets(s, False)),
    ),
    loss.CrossEntropyLoss: lambda s: (loss.CrossEntropyLoss(), (_outputs(s, C), _targets(s))),
    loss.DeepSVDDLoss: lambda s: (
        loss.DeepSVDDLoss(n_dim=D),
        (_outputs(s, D), _targets(s, False)),
    ),
    loss.EnergyMarginLoss: lambda s: (
        loss.EnergyMarginLoss(full_train_loss=1.0),
        (_outputs(s, C), _targets(s), nn.Linear(1, 2)),
    ),
    loss.EnergyRegularizedLoss: lambda s: (
        loss.EnergyRegularizedLoss(),
        (_outputs(s, C), _targets(s)),
    ),
    loss.EntropicOpenSetLoss: lambda s: (
        loss.EntropicOpenSetLoss(),
        (_outputs(s, C), _targets(s)),
    ),
    loss.IILoss: lambda s: (
        loss.IILoss(n_classes=C, n_embedding=D),
        (_outputs(s, D), _targets(s, False)),
    ),
    loss.LogitNorm: lambda s: (loss.LogitNorm(), (_outputs(s, C), _targets(s, False))),
    loss.MCHADLoss: lambda s: (
        loss.MCHADLoss(n_classes=C, n_dim=D),
        (_outputs(s, C).abs(), _targets(s)),
    ),
    loss.ObjectosphereLoss: lambda s: (
        loss.ObjectosphereLoss(),
        (_outputs(s, C), _outputs(s, D), _targets(s)),
    ),
    loss.OutlierExposureLoss: lambda s: (
        loss.OutlierExposureLoss(),
        (_outputs(s, C), _targets(s)),
    ),
    loss.DeepSADLoss: lambda s: (
        loss.DeepSADLoss(n_dim=D),
        (_outputs(s, D), _targets(s)),
    ),
    loss.VOSRegLoss: lambda s: (
        loss.VOSRegLoss(nn.Linear(1, 2), nn.Linear(C, 1)),
        (_outputs(s, C), _targets(s, False)),
    ),
    loss.VirtualOutlierSynthesizingRegLoss: lambda s: (
        loss.VirtualOutlierSynthesizingRegLoss(
            nn.Linear(1, 2),
            nn.Linear(C, 1),
            fc=nn.Linear(D, C),
            sample_number=4,
            sample_from=20,
        ),
        (_outputs(s, C), _outputs(s, D), _targets(s)),
    ),
}
