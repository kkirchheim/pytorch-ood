from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..api import LossInfo, Representation, Task
from ..utils import apply_reduction


def cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """
    Standard cross-entropy, but ignores OOD inputs.

    :param logits: logits of shape :math:`B \\times C` or :math:`B \\times C \\times H \\times W`
    :param targets: labels of shape :math:`B` or :math:`B \\times H \\times W`; labels :math:`< 0` mark OOD
        samples, which contribute zero loss
    :param reduction: one of ``mean``, ``sum``, ``none``; ``None`` is the same as ``none``
    :return: the loss
    """
    masked_targets = torch.where(targets < 0, -100, targets)
    loss = F.cross_entropy(logits, masked_targets, reduction="none", ignore_index=-100)
    return apply_reduction(loss, reduction=reduction)


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-entropy, but ignores OOD inputs: samples with targets :math:`< 0` contribute zero loss.
    Note that with ``reduction="mean"`` the sum is divided by the total number of samples,
    including the OOD samples.
    """

    # the standard objective; no OOD paper introduced it
    info = LossInfo(
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
        inputs={Representation.LOGITS},
        supervised=False,
    )

    def __init__(self, reduction: Optional[str] = "mean"):
        """
        :param reduction: reduction method to apply. Can be one of ``mean``, ``sum`` or ``none``
        """
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates cross-entropy.

        :param logits: logits of shape :math:`B \\times C` or :math:`B \\times C \\times H \\times W`
        :param targets: labels of shape :math:`B` or :math:`B \\times H \\times W`; labels :math:`< 0` are ignored
        :return: the loss
        """
        return cross_entropy(logits, targets, reduction=self.reduction)
