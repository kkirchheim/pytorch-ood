"""

"""
from typing import Optional

import torch.nn
from torch import Tensor

from ..model.centers import ClassCenters
from ..utils import apply_reduction, is_known


class DeepSVDDLoss(torch.nn.Module):
    """
    Deep Support Vector Data Description  (SVDD) from the paper *Deep One-Class Classification*.
    It places a center :math:`\\mu` in the output space of the model and pulls IN samples towards
    the sphere with center :math:`r` it in order to learn the common factors of intra class variance.

    The loss is defined as follows:

    .. math:: \\mathcal{L}(x) = \\max \\lbrace 0, \\lVert f(x) - \\mu \\rVert_2^2 - r^2 \\rbrace

    The distance of a point to the center can be used as outlier score.

    This is an implementation of the *One-Class Deep SVDD objective*, which implies that **the radius is not
    considered trainable and should usually be set to zero**.

    In the original paper, the center is initialized with the mean of :math:`f(x)` over the dataset before training.


    :see Paper: `ICML <http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf>`__

    .. note:: This module should be moved to the correct device before using ``forward()``
    """

    def __init__(
        self,
        n_dim: int,
        reduction: Optional[str] = "mean",
        radius: float = 0.0,
        center: Optional[Tensor] = None,
    ):
        """
        :param n_dim: dimensionality :math:`n` of the output space
        :param reduction: reduction method to apply, one of ``mean``, ``sum`` or ``none``
        :param radius: radius :math:`r`
        :param center: position of the center :math:`\\mu \\in \\mathbb{R}^n` where :math:`n` is the dimensionality of
            the output space
        """
        super(DeepSVDDLoss, self).__init__()
        self._center = ClassCenters(1, n_dim, fixed=True)
        self.radius = torch.tensor(
            radius, requires_grad=False
        )  #: radius :math:`r` of the hypersphere

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

    def distance(self, x: Tensor) -> Tensor:
        """
        :return: calculates :math:`\\lVert x - \\mu \\rVert^2 - r^2`
        """
        # squeeze class dimension
        return self._center(x).squeeze(1) - self.radius.pow(2)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        :param x: features
        :param y: target labels (either IN or OOD). If not given, will assume all samples are IN.
        :return: :math:`\\lVert x - \\mu \\rVert^2 - r^2`
        """
        loss = DeepSVDDLoss.svdd_loss(x, self.center, radius=self.radius, y=y)
        return apply_reduction(loss, self.reduction)

    @staticmethod
    def svdd_loss(
        x: Tensor, center: ClassCenters, radius: Tensor = 0.0, y: Optional[Tensor] = None
    ) -> Tensor:
        """
        Calculates the loss. Treats all IN samples equally, and ignores all OOD samples.
        If no labels are given, assumes all samples are IN.

        :param x: features
        :param center: center of sphere
        :param radius: radius of sphere
        :param y: Optional labels.
        """
        if y is not None:
            known = is_known(y)
        else:
            known = torch.ones(size=(x.shape[0],)).bool()

        loss = torch.zeros(size=(x.shape[0],)).to(x.device)

        if known.any():
            loss[known] = (center(x[known]).squeeze(1) - radius.pow(2)).relu()

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
