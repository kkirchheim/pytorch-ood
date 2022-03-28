import logging

import torch
from torch import nn

from oodtk import utils

log = logging.getLogger(__name__)


class RunningCenters(nn.Module):
    """"""

    def __init__(self, n_chasses, n_features):
        # create buffer for centers. those buffers will be updated during training, and are fixed during evaluation
        running_centers = torch.empty(
            size=(self.n_classes, self.n_embedding), requires_grad=False
        ).double()
        num_batches_tracked = torch.empty(size=(1,), requires_grad=False).double()
        self.register_buffer("running_centers", running_centers)
        self.register_buffer("num_batches_tracked", num_batches_tracked)
        self.reset_running_stats()

    @property
    def centers(self) -> torch.Tensor:
        """
        :return: current class center estimates
        """
        return self.running_centers

    def reset_running_stats(self) -> None:
        """
        Resets the running stats of online class center estimates.
        """
        log.info("Reset running stats")
        nn.init.zeros_(self.running_centers)
        nn.init.zeros_(self.num_batches_tracked)

    def calculate_centers(self, embeddings, target) -> torch.Tensor:
        mu = torch.full(
            size=(self.n_classes, self.n_embedding),
            fill_value=float("NaN"),
            device=embeddings.device,
        )
        for clazz in target.unique(sorted=False):
            mu[clazz] = embeddings[target == clazz].mean(dim=0)  # all instances of this class
        return mu

    def update(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Update running centers

        :param x: inputs
        :param target: class labels
        :return: per class mean of inputs
        """
        batch_classes = torch.unique(target, sorted=False)
        n_instances = x.shape[0]
        # calculate empirical centers
        mu = self.calculate_centers(x, target)
        # update running mean centers
        cma = mu[batch_classes] + self.running_centers[batch_classes] * self.num_batches_tracked
        self.running_centers[batch_classes] = cma / (self.num_batches_tracked + 1)
        self.num_batches_tracked += 1
        return mu

    def calculate_distances(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: input points
        :return: distances to class centers
        """
        distances = utils.pairwise_distances(self.centers, x)
        return distances

    def forward(self, x: torch.Tensor):
        """
        Calculates distances to centers
        :param x:
        :return: distances to all centers
        """
        return utils.pairwise_distances(self.centers, x)
