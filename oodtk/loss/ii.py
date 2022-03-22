import logging

import torch
import torch.nn as nn
from torch.nn import init

import oodtk.utils
from oodtk.model.centers import RunningCenters

log = logging.getLogger(__name__)


class IILoss(nn.Module):
    """
    II Loss function from *Learning a neural network based representation for open set recognition*.


    :see Paper: https://arxiv.org/pdf/1802.04365.pdf
    :see Implementation: https://github.com/shrtCKT/opennet

    .. note::
        * The device of the given embedding will be used as device for all calculations.

    .. warning::
         * We added running centers for online class center estimation. This is only an approximation and results
           might be different if the centers are actually calculated.

    """

    def __init__(self, n_classes, n_embedding):
        """
        :param n_classes: number of classes
        :param n_embedding: embedding dimensionality
        """
        super(IILoss, self).__init__()
        self.running_centers = RunningCenters(n_chasses=n_classes, n_embedding=n_embedding)

    @property
    def centers(self) -> torch.Tensor:
        """
        :return: current class center estimates
        """
        return self.running_centers.centers

    def _calculate_spreads(self, mu, x, targets) -> torch.Tensor:
        """
         Calculate sum of (squared) distances of all instances to the class center

        :param mu:
        :param x:
        :param targets:
        :return:
        """
        spreads = torch.zeros((self.n_classes,), device=x.device)

        for clazz in targets.unique(sorted=False):
            class_x = x[targets == clazz]  # all instances of this class
            spreads[clazz] = torch.norm(class_x - mu[clazz], p=2).pow(2).sum()

        return spreads

    def _get_center_distances(self, mu):
        """
        get distances of centers
        :param mu:
        :return:
        """
        dists = oodtk.utils.pairwise_distances(mu)

        # set diagonal elements to "high" value (this value will limit the inter seperation, so cluster
        # do not drift apart infinitely)
        dists[torch.eye(self.n_classes, dtype=torch.bool)] = 1e24
        return dists

    def calculate_distances(self, x):
        """

        :param x: input points
        :return: distances to class centers
        """
        return oodtk.utils.pairwise_distances(self.centers, x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class membership probability

        :param x: features
        :return: class membership probabilities
        """
        distances = self.calculate_distances(x).softmin(dim=1)

    def forward(self, x, target) -> torch.Tensor:
        """

        :param x: embeddings of samples
        :param target: label of samples
        """
        batch_classes = torch.unique(target, sorted=False)

        if self.training:
            # calculate empirical centers
            mu = self.running_centers.update(x, target)  # self._calculate_centers(x, target)
        else:
            # when testing, use the running empirical class centers
            mu = self.running_centers.centers

        # calculate sum of class spreads and divide by the number of instances
        intra_spread = self._calculate_spreads(mu, x, target).sum() / x.shape[0]

        # calculate distance between all (present) class centers
        dists = self._get_center_distances(mu[batch_classes])

        # the minimum distance between all class centers is the inter separation
        inter_separation = -torch.min(dists)

        # intra_spread should be minimized, inter_separation maximized
        # we substract the margin from the inter seperation, so the overall loss will always be > 0.
        # this does not influence on the results of the loss, because constant offsets have no impact on the gradient.
        return intra_spread, inter_separation
