import torch
from torch import nn
from torch.nn import functional as F

from oodtk.utils import is_known


def cross_entropy(x, targets):
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """
    known = is_known(targets)
    if not known.any():
        return torch.zeros(size=(1,))

    return F.cross_entropy(x[known], targets[known])


class CrossEntropy(nn.Module):
    """
    Standard Cross-entropy, but ignores OOD inputs.
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, targets) -> torch.Tensor:
        cross_entropy(x, targets)
