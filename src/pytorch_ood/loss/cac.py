"""
CACLoss
----------------------------------------------

..  automodule:: pytorch_ood.nn.loss.cac
    :members: cac_rejection_score, CACLoss

"""
import torch as torch
import torch.nn as nn

#
from torch.nn import functional as F

from ..model.centers import ClassCenters
from ..utils import is_known


class CACLoss(nn.Module):
    """
    Class Anchor Clustering Loss from
    *Class Anchor Clustering: a Distance-based Loss for Training Open Set Classifiers*

    :see Paper: https://arxiv.org/abs/2004.02434
    :see Implementation: https://github.com/dimitymiller/cac-openset/

    """

    def __init__(self, n_classes, magnitude=1, alpha=1):
        """
        Centers are initialized as unit vectors, scaled by the magnitude.

        :param n_classes: number of classes
        :param magnitude: magnitude of class anchors
        :param alpha: weight for anchor term
        """
        super(CACLoss, self).__init__()
        self.n_classes = n_classes
        self.magnitude = magnitude
        self.alpha = alpha
        # anchor points are fixed, so they do not require gradients
        self.centers = ClassCenters(n_classes, n_classes, fixed=True)
        self._init_centers()

    def _init_centers(self):
        """Init anchors with 1, scale by"""
        nn.init.eye_(self.centers.params)
        self.centers.params.data *= self.magnitude  # scale with magnitude

    def forward(self, distances, target) -> torch.Tensor:
        """
        :param distances: distance matrix
        :param target: labels for samples
        """
        assert distances.shape[1] == self.n_classes

        known = is_known(target)
        if known.any():
            target_known = target[known]
            d_known = distances[known]
            len_dist_known = len(d_known)

            d_true = torch.gather(input=d_known, dim=1, index=target_known.view(-1, 1)).view(-1)
            anchor_loss = d_true.mean()

            non_target = torch.arange(
                0, self.n_classes - 1, dtype=torch.long, device=distances.device
            ).expand(d_known.shape[0], self.n_classes - 1)
            is_last_class = target_known == self.n_classes
            non_target[is_last_class, target_known[is_last_class]] = self.n_classes - 1

            d_other = torch.gather(d_known, dim=1, index=non_target)
            # for numerical stability, we clamp the distance values
            tuplet_loss = (-d_other + d_true.unsqueeze(1)).clamp(max=50).exp()
            tuplet_loss = torch.log(1 + tuplet_loss.sum(dim=1)).mean()
        else:
            anchor_loss = torch.tensor(0.0, device=distances.device)
            tuplet_loss = torch.tensor(0.0, device=distances.device)

        return self.alpha * anchor_loss + tuplet_loss

    def calculate_distances(self, x):
        """

        :param x: input points
        :return: distances to class centers
        """
        return self.centers(x).pow(2)

    @staticmethod
    def score(distance):
        """
        Rejection score used by the CAC loss

        :param distance: distance of instances to class centers
        :return:
        """
        scores = distance * (1 - F.softmin(distance, dim=1))
        return scores.max(dim=1).values
