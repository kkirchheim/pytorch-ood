import torch
from torch import nn
from torch.nn import functional as F

from ..utils import apply_reduction, is_known


def cross_entropy(x, targets, reduction="mean"):
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """
    loss = torch.zeros(x.shape[0], device=x.device)
    known = is_known(targets)

    if known.any():
        loss[known] = F.cross_entropy(x[known], targets[known], reduction="none")

    return apply_reduction(loss, reduction=reduction)


class CrossEntropy(nn.Module):
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """

    def __init__(self, reduction="mean"):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, targets) -> torch.Tensor:
        return cross_entropy(x, targets, reduction=self.reduction)
