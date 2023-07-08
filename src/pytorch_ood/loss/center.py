import logging

import torch
import torch.nn as nn

from ..model.centers import ClassCenters
from ..utils import is_known

log = logging.getLogger(__name__)


class CenterLoss(nn.Module):
    """
    Generalized version of the Center Loss from the Paper
    *A Discriminative Feature Learning Approach for Deep Face Recognition*.
    For each class, this loss places a center :math:`\\mu_y` in the output space and draws representations of samples
    to their corresponding class centers, up to a radius :math:`r`.

    Calculates

    .. math::
        \\mathcal{L}(x,y) = \\max \\lbrace  d(f(x),\\mu_y) - r , 0 \\rbrace

    where :math:`d` is some measure of dissimilarity, like the squared distance.

    With radius :math:`r=0` and the squared euclidean distance as :math:`d(\\cdot,\\cdot)`, this is equivalent to
    the original center loss, which is also referred to as the *soft-margin loss* in some publications.

    :see Implementation: `GitHub <https://github.com/KaiyangZhou/pytorch-center-loss>`__
    :see Paper: `ECCV 2016 <https://ydwen.github.io/papers/WenECCV16.pdf>`__
    """

    def __init__(
        self,
        n_classes: int,
        n_dim: int,
        magnitude: float = 1.0,
        radius: float = 0.0,
        fixed: bool = False,
    ):
        """
        :param n_classes: number of classes :math:`C`
        :param n_dim: dimensionality of center space :math:`D`
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
            torch.nn.init.eye_(self._centers._params)
            if not self._centers._params.requires_grad:
                self._centers._params.mul_(self.magnitude)
        # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
        # different then the number of classes
        # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers.params)
            if self.magnitude != 1:
                log.warning("Not applying magnitude parameter.")

    def forward(self, distmat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss. Ignores OOD inputs.

        :param distmat: matrix of distances of each point to each center with shape :math:`B \\times C`.
        :param target: ground truth labels with shape (batch_size).
        :returns: the loss values
        """
        known = is_known(target)

        if known.any():
            distmat = distmat[known]
            target = target[known]
            batch_size = distmat.size(0)

            classes = torch.arange(self.num_classes).long().to(distmat.device)
            target = target.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = target.eq(classes.expand(batch_size, self.num_classes))
            dist = (distmat - self.radius).relu() * mask.float()
            loss = dist.clamp(min=1e-12, max=1e12).mean()
        else:
            loss = torch.tensor(0.0, device=distmat.device)

        return loss
