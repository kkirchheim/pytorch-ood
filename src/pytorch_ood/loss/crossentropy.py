from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import apply_reduction, is_known


def cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """
    Standard cross-entropy, but ignores OOD inputs.
    """
    masked_targets = torch.where(targets < 0, -100, targets)
    loss = F.cross_entropy(logits, masked_targets, reduction="none", ignore_index=-100)
    return apply_reduction(loss, reduction=reduction)


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """

    def __init__(self, reduction: Optional[str] = "mean"):
        """
        :param reduction: reduction method to apply. Can be one of ``mean``, ``sum`` or ``none``
        """
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates cross-entropy.

        :param logits: logits
        :param targets: labels
        """
        return cross_entropy(logits, targets, reduction=self.reduction)
