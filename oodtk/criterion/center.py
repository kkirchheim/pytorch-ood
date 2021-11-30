import logging

import torch
import torch.nn as nn

import oodtk.utils

log = logging.getLogger(__name__)


class CenterLoss(nn.Module):
    """
    Center Loss.

    :param n_classes: number of classes.
    :param n_embedding: feature dimension.
    :param magnitude: magnitude of

    :see Implementation: https://github.com/KaiyangZhou/pytorch-center-loss
    :see Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(self, n_classes, n_embedding, magnitude=1, fixed=False):
        super(CenterLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_embedding
        self.magnitude = magnitude

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if fixed:
            self.centers.requires_grad = False

        # In the published code, they initialize centers randomly.
        # This, however, is not a good idea if the loss is used without an additional inter-class-discriminability term
        self._init_centers()

    def _init_centers(self):
        if self.num_classes == self.feat_dim:
            torch.nn.init.eye_(self.centers)

            if not self.centers.requires_grad:
                self.centers.mul_(self.magnitude)

            # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
            # different then the number of classes
            # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers)

            if self.magnitude != 1:
                log.warning(f"Not applying magnitude parameter.")

    def forward(self, x, labels) -> torch.Tensor:
        """
        :param: x: feature matrix with shape (batch_size, feat_dim).
        :param labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss

    def calculate_distances(self, embeddings):
        distances = oodtk.utils.torch_get_squared_distances(self.centers, embeddings)
        return distances

    def predict(self, embeddings):
        distances = self.calculate_distances(embeddings)
        return nn.functional.softmin(distances, dim=1)
