import logging

import torch
from torch import nn

from .. import utils

log = logging.getLogger(__name__)


class ClassCenters(nn.Module):
    """
    Several methods for OOD Detection propose to model a center :math:`\\mu_y` for each class.
    These centers are either static, or learned via gradient descent.

    The centers are also known as class proxy, class prototype or class anchor.
    """

    def __init__(self, n_classes: int, n_features: int, fixed: bool = False):
        """

        :param n_classes: number of classes vectors
        :param n_features: dimensionality of the space in which the centers live
        :param fixed: False if the centers should be learnable parameters, True if they should be fixed at their
            initial position
        """
        super(ClassCenters, self).__init__()
        # anchor points are fixed, so they do not require gradients
        self._params = nn.Parameter(torch.zeros(size=(n_classes, n_features)))

        if fixed:
            self._params.requires_grad = False

    @property
    def num_classes(self) -> int:
        return self.params.shape[0]

    @property
    def n_features(self) -> int:
        return self.params.shape[1]

    @property
    def params(self) -> nn.Parameter:
        """
        Class centers :math:`\\mu`
        """
        return self._params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: samples
        :returns: pairwise distance of samples to each center
        """
        assert x.shape[1] == self.n_features
        return utils.pairwise_distances(x, self.params)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make class membership predictions based on the softmin of the distances to each center.

        :param x: embeddings of samples
        :returns: normalized pairwise distance of samples to each center
        """
        distances = utils.pairwise_distances(x, self.params)
        return nn.functional.softmin(distances, dim=1)


class RunningCenters(nn.Module):
    """
    Estimates class centers from batches of data using a running mean estimator.
    """

    def __init__(self, n_classes, n_embedding):
        """

        :param n_classes:
        :param n_embedding:
        """
        super(RunningCenters, self).__init__()
        self.num_classes = n_classes
        self.n_embedding = n_embedding
        # create buffer for centers. those buffers will be updated during training, and are fixed during evaluation
        running_centers = torch.empty(
            size=(self.num_classes, self.n_embedding), requires_grad=False
        ).float()
        num_batches_tracked = torch.empty(size=(1,), requires_grad=False).float()
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
            size=(self.num_classes, self.n_embedding),
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
        return utils.pairwise_distances(x, self.centers)
