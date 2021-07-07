
import torch as torch
import torch.nn as nn
#
from torch.nn import functional as F

import oodtk.utils


class CACLoss(nn.Module):
    """
    Class Anchor Clustering Loss from
    *Class Anchor Clustering: a Distance-based Loss for Training Open Set Classifiers*


    Centers are initialized as unit vectors, scaled by the magnitude.


    :param n_classes: number of classes, equald number of class centers
    :param magnitude: magnitude of class anchors
    :param lmbda: weight :math:`\lambda` for loss terms


    :see Paper: https://arxiv.org/abs/2004.02434
    :see Implementation: https://github.com/dimitymiller/cac-openset/

    """

    def __init__(self, n_classes, magnitude, lmbda):

        super(CACLoss, self).__init__()
        self.n_classes = n_classes
        self.magnitude = magnitude
        self.lambda_ = lmbda

        # anchor points are fixed, so they do not require gradients
        self.anchors = nn.Parameter(torch.zeros(size=(n_classes, n_classes)))
        self._init_centers()

    @property
    def centers(self):
        """
        Class centers, a.k.a. Anchors
        """
        return self.anchors

    def _init_centers(self):
        """Init anchors with 1, scale by """
        nn.init.eye_(self.anchors)
        self.anchors.requires_grad = False
        self.anchors *= self.magnitude  # scale with magnitude

    def forward(self, embedding, target) -> torch.Tensor:
        """
        :param embedding: embeddings of samples
        :param target: labels for samples
        """
        assert (embedding.shape[1] == self.n_classes)

        distances = self.calculate_distances(embedding)
        d_true = torch.gather(input=distances, dim=1, index=target.view(-1, 1)).view(-1)
        anchor_loss = d_true.mean()

        # calc distances to all non_target tensors
        tmp = [[i for i in range(self.n_classes) if target[x] != i] for x in range(len(distances))]
        non_target = torch.Tensor(tmp).long().to(embedding.device)
        d_other = torch.gather(distances, 1, non_target)

        # for numerical stability, we clamp the distance values
        tuplet_loss = (-d_other + d_true.unsqueeze(1)).clamp(max=50).exp()  # torch.exp()
        tuplet_loss = torch.log(1 + torch.sum(tuplet_loss, dim=1)).mean()

        return self.lambda_ * anchor_loss, tuplet_loss

    def calculate_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        :param embeddings: embeddings of samples
        :returns: squared euclidean distance of embeddings to anchors
        """
        distances = oodtk.utils.torch_get_squared_distances(self.anchors, embeddings)
        return distances

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Make class membership predictions

        :param embeddings: embeddings of samples
        """
        distances = self.calculate_distances(embeddings)
        return nn.functional.softmin(distances, dim=1)


def rejection_score(distance):
    """
    Rejection score used by the CAC loss

    :param distance:
    :return:
    """
    scores = distance * (1 - F.softmin(distance, dim=1))
    return scores
