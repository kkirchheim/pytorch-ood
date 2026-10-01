""" """

import torch.nn
import torch.nn.functional as F

from ..api import LossInfo, Paper, Representation, Task
from ..utils import is_unknown


class BackgroundClassLoss(torch.nn.Module):
    """
    The idea of the background-class is that OOD samples are mapped to an individual class during training.
    This implementation uses the normal cross-entropy, but handles remapping of the background class labels
    to positive target labels.
    Thus, when the target labels are :math:`\\lbrace 0, 1, 2, ..., N - 1 \\rbrace`
    we will remap all entries with target label :math:`<0` to :math:`N`.

    The networks output layer has to include :math:`N+1` outputs, so logits are
    in the shape  :math:`B \\times (N + 1)`.
    """

    info = LossInfo(
        paper=Paper(
            title="Reducing Network Agnostophobia",
            venue="NeurIPS",
            year=2018,
            url="https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf",
            code=None,
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
        inputs={Representation.LOGITS},
        supervised=True,
    )

    def __init__(self, n_classes: int, reduction: str = "mean"):
        """
        :param n_classes: number of classes :math:`N` (not counting background class)
        :param reduction: can be one of ``none``, ``mean``, ``sum``
        """
        super(BackgroundClassLoss, self).__init__()
        self.num_classes = n_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param logits: class logits
        :param targets: target labels

        :return: Cross-Entropy for remapped samples
        """
        if (targets >= self.num_classes).any():
            raise ValueError(f"Target label to large: {targets.max()}")

        # remap outliers to the background class, without changing the caller's tensor
        targets = torch.where(
            is_unknown(targets), torch.full_like(targets, self.num_classes), targets
        )

        return F.cross_entropy(logits, targets, reduction=self.reduction)
