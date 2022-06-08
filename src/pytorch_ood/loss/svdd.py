"""

"""
from typing import Optional

import torch.nn

from ..model.centers import ClassCenters
from ..utils import apply_reduction, is_known


class DeepSVDDLoss(torch.nn.Module):
    """
    Deep Support Vector Data Description from the paper *Deep One-Class Classification*.
    It models a center :math:`\\mu` in the output space of the model and pulls IN samples towards it in order
    to learn the common factors of intra class variance.

    This distance to this center can be used as outlier score.

    In the original paper, the center is initialized with the mean of :math:`f(x)` over the dataset before training.


    .. math:: \\mathcal{L}(x) = \\max \\lbrace 0, \\lVert \\mu - f(x) \\rVert_2^2 \\rbrace

    :see Paper: http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf

    """

    def __init__(self, n_features: int, reduction: Optional[str] = "mean"):
        """
        :param n_features: dimensionality of the output space
        :param reduction: reduction method to apply
        """
        super(DeepSVDDLoss, self).__init__()
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
        :param y: target labels (either IN or OOD)
        :return: squared distance to center
        """
        loss = DeepSVDDLoss.svdd_loss(x, y, self.center)
        known = is_known(y)
        if known.any():
            loss = self._center(x[known]).squeeze(1).pow(2)
        else:
            loss = torch.zeros(size=(x.shape[0],))

        return apply_reduction(loss, self.reduction)

    @staticmethod
    def svdd_loss(x, y, center):
        known = is_known(y)
        if known.any():
            loss = center(x[known]).squeeze(1).pow(2)
        else:
            loss = torch.zeros(size=(x.shape[0],))

        return loss


class SSDeepSVDDLoss(torch.nn.Module):
    """
    Semi-Supervised generalization of Deep Support Vector Data Description.
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
        super(DeepSVDDLoss, self).__init__()
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
        if known.any():
            loss_in = self._center(x[known]).squeeze(1).pow(2)
        else:
            loss_in = torch.zeros(size=(x.shape[0],))

        # TODO
        if (~known).any():
            loss_out = torch.zeros()
        else:
            loss_out = torch.zeros(size=(x.shape[0],))

        loss = loss_in + loss_out
        return apply_reduction(loss, self.reduction)
