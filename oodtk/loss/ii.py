import logging

import torch
import torch.nn as nn
from torch.nn import init

import oodtk.utils

log = logging.getLogger(__name__)


class IILoss(nn.Module):
    """
    II Loss function from *Learning a neural network based representation for open set recognition*.

    :param n_classes: number of classes
    :param n_embedding: embedding dimensionality

    :see Paper: https://arxiv.org/pdf/1802.04365.pdf
    :see Implementation: https://github.com/shrtCKT/opennet

    .. note::
        * We added running centers for online class center estimation
        * The device of the given embedding will be used as device for all calculations.


    .. warning::
        * Hassen et al. report that batch-norm bounds the output in a hypercube and thus limits the spread of distances,
          however, we could not reproduce this behavior in our experiments.

    """

    def __init__(self, n_classes, n_embedding):

        super(IILoss, self).__init__()
        self.n_classes = n_classes
        self.n_embedding = n_embedding

        # create buffer for centers. those buffers will be updated during training, and are fixed during evaluation
        running_centers = torch.empty(
            size=(self.n_classes, self.n_embedding), requires_grad=False
        ).double()
        num_batches_tracked = torch.empty(size=(1,), requires_grad=False).double()

        self.register_buffer("running_centers", running_centers)
        self.register_buffer("num_batches_tracked", num_batches_tracked)
        self.reset_running_stats()

    @property
    def centers(self):
        """
        :return: current class center estimates
        """
        return self.running_centers

    def reset_running_stats(self):
        log.info("Reset running stats")
        init.zeros_(self.running_centers)
        init.zeros_(self.num_batches_tracked)

    def calculate_centers(self, embeddings, target):
        mu = torch.full(
            size=(self.n_classes, self.n_embedding),
            fill_value=float("NaN"),
            device=embeddings.device,
        )

        for clazz in target.unique(sorted=False):
            mu[clazz] = embeddings[target == clazz].mean(dim=0)  # all instances of this class

        return mu

    def calculate_spreads(self, mu, x, y):
        spreads = torch.zeros((self.n_classes,), device=x.device)

        # calculate sum of (squared) distances of all instances to the class center
        for clazz in y.unique(sorted=False):
            class_x = x[y == clazz]  # all instances of this class
            spreads[clazz] = torch.norm(class_x - mu[clazz], p=2).pow(2).sum()

        return spreads

    def _get_center_distances(self, mu):
        """
        get distances of centers
        :param mu:
        :return:
        """
        dists = oodtk.utils.pairwise_distances(mu)

        # n_centers = mu.shape[0]
        # a = mu.unsqueeze(1).expand(n_centers, n_centers, mu.size(1)).double()
        # b = mu.unsqueeze(0).expand(n_centers, n_centers, mu.size(1)).double()
        # dists = torch.norm(a - b, p=2, dim=2).pow(2)

        # set diagonal elements to "high" value (this value will limit the inter seperation, so cluster
        # do not drift apart infinitely)
        dists[torch.eye(self.n_classes, dtype=torch.bool)] = 1e24
        return dists

    def calculate_distances(self, x):
        """

        :param x: input points
        :return: distances to class centers
        """
        distances = oodtk.utils.pairwise_distances(self.centers, x)
        return distances

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class membership probability

        :param x: features
        :return: class membership probabilities
        """
        distances = self.calculate_distances(x)
        return nn.functional.softmin(distances, dim=1)

    def forward(self, x, target) -> torch.Tensor:
        """

        :param x: embeddings of samples
        :param target: label of samples
        """
        batch_classes = torch.unique(target, sorted=False)
        n_instances = x.shape[0]

        if self.training:
            # calculate empirical centers
            mu = self.calculate_centers(x, target)

            # update running mean centers
            cma = (
                mu[batch_classes] + self.running_centers[batch_classes] * self.num_batches_tracked
            )
            self.running_centers[batch_classes] = cma / (self.num_batches_tracked + 1)
            self.num_batches_tracked += 1
        else:
            # when testing, use the running empirical class centers
            mu = self.running_centers

        # calculate sum of class spreads and divide by the number of instances
        intra_spread = self.calculate_spreads(mu, x, target).sum() / n_instances

        # calculate distance between all (present) class centers
        dists = self._get_center_distances(mu[batch_classes])

        # the minimum distance between all class centers is the inter separation
        inter_separation = -torch.min(dists)

        # intra_spread should be minimized, inter_separation maximized
        # we substract the margin from the inter seperation, so the overall loss will always be > 0.
        # this does not influence on the results of the loss, because constant offsets have no impact on the gradient.
        return intra_spread, inter_separation
