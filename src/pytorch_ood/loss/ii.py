import logging

import torch
import torch.nn as nn
from torch.nn.functional import softmin

from ..model.centers import RunningCenters
from ..utils import is_known, pairwise_distances

log = logging.getLogger(__name__)


def _get_center_distances(mu: torch.Tensor, eps: float = 1e24) -> torch.Tensor:
    """
    Get distances of centers

    :param mu: centers
    :param eps: very large values to for diagonal entries
    :return: distance matrix
    """
    dists = pairwise_distances(mu)
    # set diagonal elements to "high" value (this value will limit the inter separation, so cluster
    # do not drift apart infinitely)
    dists[torch.eye(len(mu), dtype=torch.bool)] = eps
    return dists


class IILoss(nn.Module):
    """
    II Loss function from *Learning a neural network based representation for open set recognition*.


    :see Paper: `ArXiv <https://arxiv.org/pdf/1802.04365.pdf>`__
    :see Implementation: `GitHub <https://github.com/shrtCKT/opennet>`__

    .. warning::
         * We added running centers for online class center estimation. This is only an approximation and results
           might be different if the centers are actually calculated as described in the paper.
           However, this enables better estimation of the performance during training, without having calculate
           the centers over the entire dataset. Empirically, we found that these centers work well.

    """

    def __init__(self, n_classes: int, n_embedding: int, alpha: float = 1.0):
        """
        :param n_classes: number of classes
        :param n_embedding: embedding dimensionality
        :param alpha: weight for both loss terms
        """
        super(IILoss, self).__init__()
        self.num_classes = n_classes
        self.running_centers = RunningCenters(n_classes=n_classes, n_embedding=n_embedding)
        self.alpha = alpha

    @property
    def centers(self) -> RunningCenters:
        """
        :return: current class center estimates
        """
        return self.running_centers

    def _calculate_spreads(
        self, mu: torch.Tensor, x: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
         Calculate sum of (squared) distances of all instances to the class center

        :param mu: centers
        :param x: embeddings
        :param targets: target labels
        :return: sum of squared distance to centers
        """
        spreads = torch.zeros((self.num_classes,), device=x.device)
        for clazz in targets.unique(sorted=False):
            class_x = x[targets == clazz]  # all instances of this class
            spreads[clazz] = torch.norm(class_x - mu[clazz], p=2).pow(2).sum()
        return spreads

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: embeddings
        :return: distances matrix with distances to class centers in output space
        """
        return pairwise_distances(x, self.centers.centers)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class membership probability

        :param x: embeddings
        :return: class membership probabilities
        """
        return softmin(self.distance(x), dim=1)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Updates running centers

        :param x: embeddings of samples
        :param target: label of samples
        """
        known = is_known(target)

        if known.any():
            batch_classes = torch.unique(target[known], sorted=False)
            if self.training:
                # calculate empirical centers
                mu = self.running_centers.update(
                    x[known], target[known]
                )  # self._calculate_centers(x, target)
            else:
                # when testing, use the running empirical class centers
                mu = self.running_centers.centers

            # calculate sum of class spreads and divide by the number of instances
            intra_spread = (
                self._calculate_spreads(mu, x[known], target[known]).sum() / x[known].shape[0]
            )
            # calculate distance between all (present) class centers
            dists = _get_center_distances(mu[batch_classes])
            # the minimum distance between all class centers is the inter separation
            inter_separation = -torch.min(dists)
            # intra_spread should be minimized, inter_separation maximized
            return intra_spread + self.alpha * inter_separation
        else:
            return torch.zeros(size=(1,))
