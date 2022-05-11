"""

"""
from typing import Optional

import torch.nn

from pytorch_ood.model.centers import ClassCenters
from pytorch_ood.utils import is_known

from ._utils import apply_reduction


class DeepSVDD(torch.nn.Module):
    """
    Deep Support Vector Data Description
    """

    def __init__(self, n_features: int, reduction: Optional[str] = "mean"):
        """
        :param n_features: dimensionality of the output space
        """
        super(DeepSVDD, self).__init__()
        self._center = ClassCenters(1, n_features, fixed=True)
        self.reduction = reduction

    @property
    def center(self) -> ClassCenters:
        """
        The center :math:`\\mu`
        """
        return self._center

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: features
        :param y: target labels
        :return:
        """
        known = is_known(y)
        loss = self._center(x[is_known(y)]).squeeze(1).pow(2)
        return apply_reduction(loss, self.reduction)
