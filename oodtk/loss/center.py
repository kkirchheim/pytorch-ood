import logging

import torch
import torch.nn as nn

import oodtk.utils
from oodtk.model.centers import ClassCenters

log = logging.getLogger(__name__)


class CenterLoss(nn.Module):
    """
    Center Loss.

    :see Implementation: https://github.com/KaiyangZhou/pytorch-center-loss
    :see Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(self, n_classes, n_embedding, magnitude=1, fixed=False):
        """
        :param n_classes: number of classes.
        :param n_embedding: feature dimension.
        :param magnitude:
        :param fixed: false if centers should be learnable
        """
        super(CenterLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_embedding
        self.magnitude = magnitude
        self.centers = ClassCenters(n_classes=n_classes, n_features=n_embedding, fixed=fixed)
        self._init_centers()

    def _init_centers(self):
        # In the published code, they initialize centers randomly.
        # However, this might bot be optimal if the loss is used without an additional inter-class-discriminability term
        if self.num_classes == self.feat_dim:
            torch.nn.init.eye_(self.centers.centers)
            if not self.centers.centers.requires_grad:
                self.centers.centers.mul_(self.magnitude)
        # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
        # different then the number of classes
        # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers.centers)
            if self.magnitude != 1:
                log.warning("Not applying magnitude parameter.")

    def forward(self, distmat, labels) -> torch.Tensor:
        """
        :param z: embeddings of samples with shape (batch_size, feat_dim).
        :param labels: ground truth labels with shape (batch_size).
        """
        batch_size = distmat.size(0)

        # distmat = (
        #     torch.pow(z, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        #     + torch.pow(self.centers.centers, 2)
        #     .sum(dim=1, keepdim=True)
        #     .expand(self.num_classes, batch_size)
        #     .t()
        # )
        # # distmat.addmm_(1, -2, z, self._centers.t())
        classes = torch.arange(self.num_classes).long().to(distmat.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).mean()
        return loss
