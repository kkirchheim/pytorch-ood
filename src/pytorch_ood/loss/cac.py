"""Class Anchor Clustering loss."""

import torch as torch
import torch.nn as nn

#
from torch.nn import functional as F

from ..api import LossInfo, Paper, Representation, Task
from ..model.centers import ClassCenters
from ..utils import is_known


class CACLoss(nn.Module):
    """
    Class Anchor Clustering Loss from the paper
    *Class Anchor Clustering: a Loss for Distance-based Open Set Recognition*.

    They place a class conditional center (called anchor) in the output space of the model and pull representations of
    points of a class :math:`y` towards the corresponding center :math:`\\mu_y` during training.
    The centers are initialized as unit vectors scaled by a magnitude and not trainable.

    With the squared distances :math:`d_c(x) = \\lVert f(x) - \\mu_c \\rVert_2^2` to the centers, the loss is

    .. math::
        \\mathcal{L}(x, y) = \\alpha \\, d_y(x) + \\log\\Bigl(1 + \\sum_{c \\neq y} e^{d_y(x) - d_c(x)}\\Bigr)

    where :math:`\\alpha` weights the anchor term.
    Samples with labels :math:`< 0` are ignored.

    They also propose an outlier score based on the distance which is implemented in the :meth:`CACLoss.score` method.

    .. rubric:: Examples

    See the :doc:`gallery example </auto_examples/loss/unsupervised/cac>`.
    """

    info = LossInfo(
        paper=Paper(
            title="Class Anchor Clustering: a Loss for Distance-based Open Set Recognition",
            venue="WACV",
            year=2021,
            url="https://arxiv.org/abs/2004.02434",
            code="https://github.com/dimitymiller/cac-openset/",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.DISTANCES},
        supervised=False,
    )

    def __init__(self, n_classes: int, magnitude: float = 1.0, alpha: float = 1.0):
        """
        Centers are initialized as unit vectors, scaled by the magnitude.

        :param n_classes: number of classes :math:`C`
        :param magnitude: magnitude of class anchors
        :param alpha: :math:`\\alpha` weight for anchor term
        """
        super(CACLoss, self).__init__()
        self.n_classes = n_classes
        self.magnitude = magnitude
        self.alpha = alpha
        # anchor points are fixed, so they do not require gradients
        self._centers = ClassCenters(n_classes, n_classes, fixed=True)
        self._init_centers()

    @property
    def centers(self) -> ClassCenters:
        """
        The class centers :math:`\\mu_y`.
        """
        return self._centers

    def _init_centers(self) -> None:
        """Init anchors as unit vectors, scale by magnitude"""
        nn.init.eye_(self.centers.params)
        self.centers.params.data *= self.magnitude

    def forward(self, distances: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the CAC loss, based on the given distance matrix and target labels.
        OOD inputs will be ignored.

        :param distances:  matrix of distances of each point to each center with shape :math:`B \\times C`.
        :param target: labels of shape :math:`B`; labels :math:`< 0` are ignored
        :return: scalar loss
        """
        assert distances.shape[1] == self.n_classes

        known = is_known(target)
        if known.any():
            target_known = target[known]
            d_known = distances[known]

            d_true = torch.gather(input=d_known, dim=1, index=target_known.view(-1, 1)).view(-1)
            anchor_loss = d_true.mean()

            non_target = torch.arange(
                0, self.n_classes - 1, dtype=torch.long, device=distances.device
            ).expand(d_known.shape[0], self.n_classes - 1)

            # required in newer versions of torch, before advances indexing
            non_target = non_target.clone()

            is_last_class = target_known == self.n_classes
            non_target[is_last_class, target_known[is_last_class]] = self.n_classes - 1

            d_other = torch.gather(d_known, dim=1, index=non_target)
            # for numerical stability, we clamp the distance values
            tuplet_loss = (-d_other + d_true.unsqueeze(1)).clamp(max=50).exp()
            tuplet_loss = torch.log(1 + tuplet_loss.sum(dim=1)).mean()
        else:
            anchor_loss = torch.tensor(0.0, device=distances.device)
            tuplet_loss = torch.tensor(0.0, device=distances.device)

        return self.alpha * anchor_loss + tuplet_loss

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: embeddings of shape :math:`B \\times C` (the dimensionality of the center space is the
            number of classes)
        :return: matrix with squared distances from each point to each center with shape :math:`B \\times C`.
        """
        return self.centers(x)

    @staticmethod
    def score(distance: torch.Tensor) -> torch.Tensor:
        """
        Rejection score proposed in the paper.

        :param distance: squared distances to the class centers, shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`.
            Computes :math:`-\\max_c d_c (1 - \\mathrm{softmin}(d)_c)`; the sign is flipped relative to the
            paper's rejection score
        """
        scores = distance * (1 - F.softmin(distance, dim=1))
        return -scores.max(dim=1).values
