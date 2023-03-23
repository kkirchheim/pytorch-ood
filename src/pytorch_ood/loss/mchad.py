"""

"""
import torch
from torch import nn

from pytorch_ood.model import ClassCenters

from ..utils import apply_reduction, is_unknown
from .center import CenterLoss
from .crossentropy import CrossEntropyLoss


class MCHADLoss(nn.Module):
    """
    Implements the MCHAD loss from the Paper *Multi-Class Hypersphere Anomaly-Detection*.

    The Loss places a center :math:`\\mu_y` for each class :math:`y` in the output space of the model and
    has three components:

    .. math::
        \\mathcal{L}_{\\Lambda}(x,y) = \\max  \\lbrace 0, \\Vert \\mu_y - f(x)_y \\Vert^2_2 - r^2 \\rbrace

        \\mathcal{L}_{\\Delta}(x,y) = \\log(1 + \\sum_{i \\neq y} e^{\\Vert \\mu_y - f(x)_y \\Vert^2_2 -  \\Vert \\mu_y - f(x)_i \\Vert^2_2} )

        \\mathcal{L}_{\\Theta}(x) = \\sum_i \\max \\lbrace 0, (r + m)^2 - \\Vert f(x) - \\mu_y \\Vert^2  \\rbrace


    Intuitively, the first term forces the samples to cluster tightly in a sphere of radius :math:`r`
    around the corresponding class centers.
    The second term  ensures that the (learnable) class centers remain separable and do not collapse.
    The third term makes sure that OOD samples have at least a distance :math:`m` to the surface of each hypersphere.

    The loss can be used in a supervised, as well as in an unsupervised manner.

    :see Implementation: `GitLab <https://gitlab.com/kkirchheim/mchad>`__
    :see Paper: `ICPR <https://ieeexplore.ieee.org/document/9956337>`__
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
        :param weight_center: weight :math:`\\lambda_{\\Lambda}` for the center loss term
        :param weight_nll: weight  :math:`\\lambda_{\\Delta}` for the maximum likelihood term
        :param weight_oe: weight  :math:`\\lambda_{\\Theta}` for the outlier exposure term
        """
        super(MCHADLoss, self).__init__()

        # loss function components: center loss, cross-entropy and regularization
        self.center_loss: CenterLoss = CenterLoss(n_classes=n_classes, n_dim=n_dim, radius=radius)
        self.nll_loss: CrossEntropyLoss = CrossEntropyLoss(reduction="mean")
        self.regu_loss: CenterRegularizationLoss = CenterRegularizationLoss(
            margin=margin, reduction="sum"
        )

        self.weight_center = weight_center
        self.weight_nll = weight_nll
        self.weight_oe = weight_oe

    @property
    def centers(self) -> ClassCenters:
        """
        Class centers :math:`\\mu_y`
        """
        return self.center_loss.centers

    def distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the distance of each embedding to each center.

        :param z: embeddings of shape :math:`B \\times D`.
        :returns: distance matrix of shape :math:`B \\times C`.
        """
        return self.centers(z)

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
            + self.weight_nll * loss_nll
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
