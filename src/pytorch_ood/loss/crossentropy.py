import warnings
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..api import LossInfo, Representation, Task
from ..utils import is_known, is_unknown


def cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, reduction: Optional[str] = "mean"
) -> torch.Tensor:
    """
    Standard cross-entropy, but ignores OOD inputs: with ``mean``, the loss is averaged over the ID
    samples (or pixels) only, and with ``none``, OOD entries are zero.

    :param logits: logits of shape :math:`B \\times C` or :math:`B \\times C \\times H \\times W`
    :param targets: labels of shape :math:`B` or :math:`B \\times H \\times W`; labels :math:`< 0` mark OOD
        samples (or pixels)
    :param reduction: one of ``mean``, ``sum``, ``none``; ``None`` is the same as ``none``
    :return: the loss; zero if there are no ID samples
    """
    if reduction is None:
        reduction = "none"
    # PyTorch would return NaN for the mean over zero samples
    if reduction == "mean" and not is_known(targets).any():
        return logits.sum() * 0.0
    masked_targets = torch.where(targets < 0, -100, targets)
    return F.cross_entropy(logits, masked_targets, reduction=reduction, ignore_index=-100)


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-entropy, but ignores OOD inputs: samples (or pixels) with targets :math:`< 0` are
    discarded, with a warning. With ``reduction="mean"``, the loss is averaged over the ID samples only.
    With ``reduction="none"``, the output keeps the shape of the targets, and OOD entries are zero.
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
        :param targets: labels of shape :math:`B` or :math:`B \\times H \\times W`; labels :math:`< 0` are discarded
        :return: the loss
        """
        if is_unknown(targets).any():
            # the other unsupervised losses warn in drop_unknown, which cannot discard pixels
            warnings.warn(
                "Discarding samples with targets < 0 (OOD), which this loss does not use.",
                stacklevel=2,
            )
        return cross_entropy(logits, targets, reduction=self.reduction)
