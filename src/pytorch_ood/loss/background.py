"""

"""
import torch.nn
import torch.nn.functional as F

from ..utils import is_unknown


class BackgroundClassLoss(torch.nn.Module):
    """
    The idea of the background-class is that OOD samples are mapped to an individual class during training.
    This implementation uses the normal cross-entropy, but handles remapping of the background class labels
    to positive target labels.
    Thus, when the target labels are :math:`\\lbrace 0, 2, ..., N - 1 \\rbrace`
    we will remap all entries with target label :math:`<0` to :math:`N`.

    The networks output layer has to include :math:`N+1` outputs, so logits are
    in the shape  :math:`B \\times (N + 1)`.
    """

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

        unknown = is_unknown(targets)
        if unknown.any():
            targets[unknown] = self.num_classes

        return F.cross_entropy(logits, targets, reduction=self.reduction)
