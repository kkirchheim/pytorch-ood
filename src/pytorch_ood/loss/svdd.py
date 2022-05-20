"""

"""
from typing import Optional

import torch.nn

from ..model.centers import ClassCenters
from ..utils import apply_reduction, is_known


class DeepSVDD(torch.nn.Module):
    """
    Deep Support Vector Data Description is a One-Class method.
    It models a center :math:`\\mu` in the output space of the model and pulls IN samples towards it in order
    to learn the common factors of intra class variance.

    This distance to this center can be used as outlier score.

    In the original paper, the center is initialized with the mean of :math:`f(x)` over the dataset before training.


    .. math:: \\mathcal{L}(x) = \\max \\lbrace 0, \\lVert \\mu - f(x) \\rVert_2^2 \\rbrace
    """

    def __init__(self, n_features: int, reduction: Optional[str] = "mean"):
        """
        :param n_features: dimensionality of the output space
        :param reduction: reduction method to apply
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
        # TODO: this will fail for single samples
        loss = self._center(x[is_known(y)]).squeeze(1).pow(2)
        return apply_reduction(loss, self.reduction)
