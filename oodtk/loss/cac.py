"""
CACLoss
----------------------------------------------

..  automodule:: oodtk.nn.loss.cac
    :members: cac_rejection_score, CACLoss

"""
import torch as torch
import torch.nn as nn

#
from torch.nn import functional as F

from oodtk.model.centers import ClassCenters
from oodtk.utils import is_known, pairwise_distances


class CACLoss(nn.Module):
    """
    Class Anchor Clustering Loss from
    *Class Anchor Clustering: a Distance-based Loss for Training Open Set Classifiers*

    :see Paper: https://arxiv.org/abs/2004.02434
    :see Implementation: https://github.com/dimitymiller/cac-openset/

    """

    def __init__(self, n_classes, magnitude=1, lambda_=1):
        """
        Centers are initialized as unit vectors, scaled by the magnitude.

        :param n_classes: number of classes
        :param magnitude: magnitude of class anchors
        :param lambda_: weight :math:`\\lambda` for loss terms
        """
        super(CACLoss, self).__init__()
        self.n_classes = n_classes
        self.magnitude = magnitude
        self.lambda_ = lambda_
        # anchor points are fixed, so they do not require gradients
        self.centers = ClassCenters(n_classes, n_classes, fixed=True)
        self._init_centers()

    def _init_centers(self):
        """Init anchors with 1, scale by"""
        nn.init.eye_(self.centers.params)
        self.centers.params *= self.magnitude  # scale with magnitude

    def forward(self, distances, target) -> torch.Tensor:
        """
        :param distances: distance matrix
        :param target: labels for samples
        """
        assert distances.shape[1] == self.n_classes

        known = is_known(target)
        if known.any():
            d_true = torch.gather(
                input=distances[known], dim=1, index=target[known].view(-1, 1)
            ).view(-1)
            anchor_loss = d_true.mean()
            # calc distances to all non_target tensors
            tmp = [
                [i for i in range(self.n_classes) if target[known][x] != i]
                for x in range(len(distances[known]))
            ]
            non_target = torch.Tensor(tmp).long().to(distances.device)
            d_other = torch.gather(distances[known], 1, non_target)
            # for numerical stability, we clamp the distance values
            tuplet_loss = (-d_other + d_true.unsqueeze(1)).clamp(max=50).exp()  # torch.exp()
            tuplet_loss = torch.log(1 + torch.sum(tuplet_loss, dim=1)).mean()
        else:
            anchor_loss, tuplet_loss = torch.tensor(0.0, device=distances.device), torch.tensor(
                0.0, device=distances.device
            )

        return self.lambda_ * anchor_loss, tuplet_loss

    def calculate_distances(self, x):
        """

        :param x: input points
        :return: distances to class centers
        """
        return self.centers(x)

    @staticmethod
    def score(distance):
        """
        Rejection score used by the CAC loss

        :param distance: distance of instances to class centers
        :return:
        """
        scores = distance * (1 - F.softmin(distance, dim=1))
        return scores
