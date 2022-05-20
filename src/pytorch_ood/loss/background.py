"""

"""
import torch.nn
import torch.nn.functional as F

from ..utils import is_unknown


class BackgroundClassLoss(torch.nn.Module):
    """
    Plain cross-entropy, but handles remapping of the background class to positive target labels.
    When the number of classes is :math:`N`, we will remap all entries with target label :math:`<0` to :math:`N`.

    The networks output layer has to include  :math:`N+1` outputs, so logits are
    in the shape  :math:`B \\times (N + 1)`.
    """

    def __init__(self, num_classes: int):
        """
        :param num_classes: number of classes (not counting background class)
        """
        super(BackgroundClassLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param logits: class logits
        :param targets: target labels

        :return: Cross-Entropy for remapped samples
        """
        if (targets >= self.num_classes).any():
            raise ValueError(f"Target label to large: {targets.max()}")

        unknown = is_unknown(targets)
        if unknown.any():
            targets[unknown] = self.num_classes

        return F.cross_entropy(logits, targets)
