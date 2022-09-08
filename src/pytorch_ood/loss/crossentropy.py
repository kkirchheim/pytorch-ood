from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import apply_reduction, is_known


def cross_entropy(
    x: torch.Tensor, targets: torch.Tensor, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """
    # known = is_known(targets)
    masked_targets = torch.where(targets < 0, -100, targets)
    loss = F.cross_entropy(x, masked_targets, reduction="none", ignore_index=-100)
    return apply_reduction(loss, reduction=reduction)


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """

    def __init__(self, reduction: Optional[str] = "mean"):
        """
        :param reduction: reduction method to apply.
        """
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ """
        return cross_entropy(x, targets, reduction=self.reduction)
