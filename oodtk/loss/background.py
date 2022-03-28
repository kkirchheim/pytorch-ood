"""

"""
import torch.nn
import torch.nn.functional as F

from oodtk.utils import is_known


class BackgroundClassLoss(torch.nn.Module):
    """
    Plain cross-entropy, but handles remapping of the background class to positive target labels.
    When the number of classes is :math:`N`, we will remap all entries with target label :math:`<0` to :math:`N`.
    """

    def __init__(self, num_classes: int):
        """
        :param num_classes: number of classes
        """
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """

        :return: cross-entropy loss
        """
        known = is_known(targets)
        targets[known] = self.num_classes
        return F.cross_entropy(logits, targets)
