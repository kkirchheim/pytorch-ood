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

    :see Paper: `Link <http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf>`__

    :note: The center is a parameter, so this model has to be moved to the correct device
    """

    def __init__(
        self, n_dim: int, reduction: Optional[str] = "mean", center: Optional[torch.Tensor] = None
    ):
        """
        :param n_dim: dimensionality of the output space
        :param reduction: reduction method to apply
        :param center: position of the center :math:`\\mu \\in \\mathbb{R}^n where :math:`n` is the dimensionality of
        the output space
        """
        super(DeepSVDDLoss, self).__init__()
        self._center = ClassCenters(1, n_dim, fixed=True)

        # initialize center values, if given
        if center is not None:
            assert center.shape == (n_dim,)
            self._center.params.data = center.reshape(1, n_dim)

        self.reduction = reduction

    @property
    def center(self) -> ClassCenters:
        """
        The center :math:`\\mu`
        """
        return self._center

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: features
        :param y: target labels (either IN or OOD). If not given, will assume IN.
        :return: squared distance to center
        """
        loss = DeepSVDDLoss.svdd_loss(x, self.center, y)
        return apply_reduction(loss, self.reduction)

    @staticmethod
    def svdd_loss(x: torch.Tensor, center: ClassCenters, y=None) -> torch.Tensor:
        """
        Calculates the loss. Treats all IN samples equally, and ignores all OOD samples.
        If no labels are given, assumes all samples are IN.
        """
        if y is not None:
            known = is_known(y)
        else:
            known = torch.ones(size=(x.shape[0],)).bool()

        loss = torch.zeros(size=(x.shape[0],)).to(x.device)

        if known.any():
            loss[known] = center(x[known]).squeeze(1).pow(2)

        return loss


class SSDeepSVDDLoss(torch.nn.Module):
    """
    Semi-Supervised generalization of Deep Support Vector Data Description.
    It places a center :math:`\\mu` in the output space of the model and pulls IN samples towards this center in order
    to learn the common factors of intra class variance.

    This distance of a representation this center can be used as outlier score for the corresponding input.

    In the original paper, the center is initialized with the mean of :math:`f(x)` over the dataset before training.
    """

    def __init__(self, n_features: int, reduction: Optional[str] = "mean"):
        """
        :param n_features: dimensionality of the output space
        :param reduction: reduction method to apply
        """
        super(SSDeepSVDDLoss, self).__init__()
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
        loss = torch.zeros(size=(x.shape[0],))

        if known.any():
            loss[known] = self._center(x[known]).squeeze(1).pow(2)

        # TODO
        if (~known).any():
            loss[~known] = 1 / self._center(x[~known]).squeeze(1).pow(2)

        return apply_reduction(loss, self.reduction)
