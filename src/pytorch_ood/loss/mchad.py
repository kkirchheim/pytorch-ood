"""

"""
import torch
from torch import nn

from ..utils import apply_reduction, is_unknown
from .center import CenterLoss
from .crossentropy import CrossEntropyLoss


class MCHADLoss(nn.Module):
    """
    From the Paper *Multi-Class Hypersphere Anomaly-Detection*.

    The loss can be used supervised as well as unsupervised.

    :see Implementation: `GitLab <https://gitlab.com/kkirchheim/mchad>`__
    """

    def __init__(
        self,
        n_classes: int,
        n_dim: int,
        radius: float = 0,
        margin: float = 0,
        weight_center: float = 1.0,
        weight_nll: float = 1.0,
        weight_oe: float = 1.0,
    ):
        """
        :param n_classes: number of classes  :math:`C`
        :param n_dim: dimensionality of the output space :math:`D`
        :param radius: radius of the hyperspheres
        :param margin: margin around hyperspheres
        :param weight_center: weight for the center loss term
        :param weight_nll: weight for the maximum likelihood term
        :param weight_oe: weight for the outlier exposure term
        """
        super(MCHADLoss, self).__init__()

        # loss function components: center loss, cross-entropy and regularization
        self.center_loss = CenterLoss(n_classes=n_classes, n_dim=n_dim, radius=radius)
        self.nll_loss = CrossEntropyLoss(reduction="mean")
        self.regu_loss = CenterRegularizationLoss(margin=margin, reduction="sum")

        self.weight_center = weight_center
        self.weight_nll = weight_nll
        self.weight_oe = weight_oe

    @property
    def centers(self):
        """
        Class centers :math:`\\mu_y`
        """
        return self.center_loss.centers

    def calculate_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the distance of each embedding to each center.

        :param z: embeddings of shape :math:`B \\times D`.
        :returns: distance metrics of shape :math:`B \\times C`.
        """
        return self.center_loss.calculate_distances(z)

    def forward(self, distmat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param distmat: distance matrix  shape :math:`B \\times C`.
        :param y: labels
        :returns: loss values
        """
        loss_center = self.center_loss(distmat, y)
        # cross-entropy with integrated softmax becomes softmin with e^-x
        loss_nll = self.nll_loss(-distmat, y)
        loss_out = self.regu_loss(distmat, y)

        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )

        return loss


class CenterRegularizationLoss(nn.Module):
    """
    Regularization Term, uses sum reduction
    """

    def __init__(self, margin: float, reduction="sum"):
        """
        :param margin: Margin around centers of the spheres (i.e. including the original radius)
        """
        super(CenterRegularizationLoss, self).__init__()
        self.margin = torch.nn.Parameter(torch.tensor([margin]).float())
        # These are fixed, so they do not require gradients
        self.margin.requires_grad = False
        self.reduction = reduction

    def forward(self, distmat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param distmat: distance matrix of samples
        :param target: target label of samples
        """
        unknown = is_unknown(target)

        if unknown.any():
            d = (self.margin.pow(2) - distmat[unknown].pow(2)).relu().sum(dim=1)

        else:
            d = torch.tensor(0.0, device=distmat.device)

        return apply_reduction(d, reduction=self.reduction)
