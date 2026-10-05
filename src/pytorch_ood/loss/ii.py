import logging

import torch
import torch.nn as nn
from torch.nn.functional import softmin

from ..api import LossInfo, Paper, Representation, Task
from ..model.centers import RunningCenters
from ..utils import drop_unknown, pairwise_distances

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

    The loss consists of the intra-class spread, the mean squared distance of the (ID) embeddings to their class
    center, and the inter-class separation, the minimum distance between the centers of the classes
    present in the batch:

    .. math::
        \\mathcal{L} = \\frac{1}{N}\\sum_i \\lVert z_i - \\mu_{y_i} \\rVert^2
        - \\alpha \\min_{j \\neq k} \\lVert \\mu_j - \\mu_k \\rVert^2

    Samples with labels :math:`< 0` are ignored.
    In evaluation mode, the stored running centers are used instead of updating them.

    .. warning::
         * We added running centers for online class center estimation. This is only an approximation and results
           might be different if the centers are actually calculated as described in the paper.
           However, this enables better estimation of the performance during training, without having calculate
           the centers over the entire dataset. Empirically, we found that these centers work well.

    .. note:: The running class centers are stored in this loss, so move it to the device of the model with
        ``.to(device)``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.loss import IILoss

        encoder = torch.nn.Linear(10, 2)  # maps inputs into a 2-dimensional embedding
        criterion = IILoss(n_classes=3, n_embedding=2)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)

        # in training mode, each batch updates the running class centers; it needs at least two classes
        x, y = torch.randn(8, 10), torch.arange(8) % 3
        loss = criterion(encoder(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # in evaluation mode, the stored centers are used
        criterion.eval()
        scores = criterion.distance(encoder(x)).min(dim=1).values.detach()  # outlier scores
    """

    info = LossInfo(
        paper=Paper(
            title="Learning a Neural-network-based Representation for Open Set Recognition",
            venue="SDM",
            year=2020,
            url="https://arxiv.org/abs/1802.04365",
            code="https://github.com/shrtCKT/opennet",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.FEATURES},
        supervised=False,
    )

    def __init__(self, n_classes: int, n_embedding: int, alpha: float = 1.0):
        """
        :param n_classes: number of classes
        :param n_embedding: embedding dimensionality
        :param alpha: weight :math:`\\alpha` of the inter-class separation term
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
        :param x: embeddings of shape :math:`B \\times D`
        :return: distances matrix of shape :math:`B \\times C` with distances to class centers in output space
        """
        return pairwise_distances(x, self.centers.centers)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class membership probability

        :param x: embeddings of shape :math:`B \\times D`
        :return: class membership probabilities of shape :math:`B \\times C`
        """
        return softmin(self.distance(x), dim=1)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Updates running centers (in training mode) and calculates the loss.
        Each batch needs samples of at least two different classes, otherwise the inter-class separation is
        not defined.

        :param x: embeddings of shape :math:`B \\times D`
        :param target: labels of shape :math:`B`; labels :math:`< 0` are ignored
        :return: scalar loss
        """
        target, x = drop_unknown(target, x)
        if len(target) == 0:
            return x.sum() * 0.0

        batch_classes = torch.unique(target, sorted=False)
        if self.training:
            # calculate empirical centers
            mu = self.running_centers.update(x, target)
        else:
            # when testing, use the running empirical class centers
            mu = self.running_centers.centers
        # calculate sum of class spreads and divide by the number of instances
        intra_spread = self._calculate_spreads(mu, x, target).sum() / x.shape[0]
        # calculate distance between all (present) class centers
        dists = _get_center_distances(mu[batch_classes])
        # the minimum distance between all class centers is the inter separation
        inter_separation = -torch.min(dists)
        # intra_spread should be minimized, inter_separation maximized
        return intra_spread + self.alpha * inter_separation
