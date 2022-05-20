import logging

import torch
import torch.nn as nn

from ..model.centers import ClassCenters
from ..utils import is_known

log = logging.getLogger(__name__)


class CenterLoss(nn.Module):
    """
    Generalized version of the *Center Loss* from the Paper
    *A Discriminative Feature Learning Approach for Deep Face Recognition*.

    Calculates

    .. math::
        \\mathcal{L}(x,y) = \\max \\lbrace  d(x,\\mu_y) - r , 0 \\rbrace

    where :math:`d` is some distance. More generally, it can be any dissimilarity function, like the squared distance,
    which is not a proper distance metric.

    Equipped with radius :math:`r=0` and the squared euclidean distance, this is also referred to as the
    *soft-margin loss* in some publications.

    :see Implementation: https://github.com/KaiyangZhou/pytorch-center-loss
    :see Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(self, n_classes, n_dim, magnitude=1, radius=0.0, fixed=False):
        """
        :param n_classes: number of classes.
        :param n_dim: dimensionality of center space
        :param magnitude:  scale :math:`\\lambda` used for center initialization
        :param radius: radius :math:`r` of spheres, lower bound for distance from center that is penalized
        :param fixed: false if centers should be learnable
        """
        super(CenterLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_dim
        self.magnitude = magnitude
        self.radius = radius
        self._centers = ClassCenters(n_classes=n_classes, n_features=n_dim, fixed=fixed)
        self._init_centers()

    @property
    def centers(self) -> ClassCenters:
        """
        :return: the :math:`\\mu` for all classes
        """
        return self._centers

    def _init_centers(self):
        # In the published code, Wen et al. initialize centers randomly.
        # However, this might bot be optimal if the loss is used without an additional
        # inter-class-discriminability term.
        # The Class Anchor Clustering initializes the centers as scaled unit vectors.
        if self.num_classes == self.feat_dim:
            torch.nn.init.eye_(self.centers.centers)
            if not self.centers.centers.requires_grad:
                self.centers.centers.mul_(self.magnitude)
        # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
        # different then the number of classes
        # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers.params)
            if self.magnitude != 1:
                log.warning("Not applying magnitude parameter.")

    def calculate_distances(self, x) -> torch.Tensor:
        """

        :param x: input points
        :return: distances to all class centers
        """
        return self.centers(x)

    def forward(self, distmat, target) -> torch.Tensor:
        """
        :param distmat: matrix of distances of samples to centers with shape (batch_size, n_centers).
        :param target: ground truth labels with shape (batch_size).
        """
        batch_size = distmat.size(0)
        known = is_known(target)

        if known.any():
            classes = torch.arange(self.num_classes).long().to(distmat.device)
            target = target.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = target.eq(classes.expand(batch_size, self.num_classes))
            dist = (distmat - self.radius).relu() * mask.float()
            loss = dist.clamp(min=1e-12, max=1e12).mean()
        else:
            loss = torch.tensor(0.0, device=distmat.device)

        return loss
