""" """

from typing import Optional

import torch.nn
from torch import Tensor

from ..api import LossInfo, Paper, Representation, Task
from ..model.centers import ClassCenters
from ..utils import apply_reduction, drop_unknown, is_known


class DeepSVDDLoss(torch.nn.Module):
    """
    Deep Support Vector Data Description  (SVDD) from the paper *Deep One-Class Classification*.
    It places a center :math:`\\mu` in the output space of the model and pulls ID samples towards
    the hypersphere of radius :math:`r` around it in order to learn the common factors of intra class variance.

    The loss is defined as follows:

    .. math:: \\mathcal{L}(x) = \\max \\lbrace 0, \\lVert f(x) - \\mu \\rVert_2^2 - r^2 \\rbrace

    The distance of a point to the center can be used as outlier score.

    With :math:`r = 0`, this is the *One-Class Deep SVDD* objective. With :math:`r > 0`, it is the
    *soft-boundary Deep SVDD* objective with a fixed radius. In the paper, the soft-boundary objective also
    optimizes the radius: after a few warm-up epochs, :math:`r` is periodically set to the
    :math:`(1 - \\nu)`-quantile of the distances :math:`\\lVert f(x) - \\mu \\rVert_2` of the training data,
    where :math:`\\nu \\in (0, 1]` bounds the fraction of training samples outside the hypersphere. To do the
    same, update ``radius`` as in the example below.

    In the original paper, the center is initialized with the mean of :math:`f(x)` over the dataset before
    training; pass it as ``center``.

    .. note:: The center :math:`\\mu` is stored in this loss, so move it to the device of the model with
        ``.to(device)``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.loss import DeepSVDDLoss

        # no bias terms, which would allow mapping every input to the center
        encoder = torch.nn.Linear(10, 2, bias=False)
        x = torch.randn(32, 10)
        # initialize the center with the mean of the initial outputs
        with torch.no_grad():
            center = encoder(x).mean(dim=0)
        criterion = DeepSVDDLoss(n_dim=2, center=center)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)

        loss = criterion(encoder(x))  # without targets, all samples are ID
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = criterion.distance(encoder(x)).detach()  # outlier scores

        # soft-boundary objective: update the radius periodically, e.g. once per epoch after a warm-up
        nu = 0.1  # bounds the fraction of training samples outside the hypersphere
        with torch.no_grad():
            distances = criterion.center(encoder(x)).squeeze(1).sqrt()
        criterion.radius.fill_(distances.quantile(1 - nu))
    """

    info = LossInfo(
        paper=Paper(
            title="Deep One-Class Classification",
            venue="ICML",
            year=2018,
            url="http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf",
            code="https://github.com/lukasruff/Deep-SVDD-PyTorch",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.FEATURES},
        supervised=False,
    )

    def __init__(
        self,
        n_dim: int,
        reduction: Optional[str] = "mean",
        radius: float = 0.0,
        center: Optional[Tensor] = None,
    ):
        """
        :param n_dim: dimensionality :math:`n` of the output space
        :param reduction: reduction method to apply, one of ``mean``, ``sum`` or ``none``
        :param radius: radius :math:`r`
        :param center: position of the center :math:`\\mu \\in \\mathbb{R}^n` where :math:`n` is the dimensionality of
            the output space
        """
        super(DeepSVDDLoss, self).__init__()
        self._center = ClassCenters(1, n_dim, fixed=True)
        # radius r of the hypersphere, a buffer so that .to() moves it
        self.register_buffer("radius", torch.tensor(radius))

        # initialize center values, if given
        if center is not None:
            assert center.shape == (n_dim,)
            self._center.params.data = center.reshape(1, n_dim)

        self.reduction = reduction

    @property
    def center(self) -> ClassCenters:
        """
        The center :math:`\\mu`
        """
        return self._center

    def distance(self, x: Tensor) -> Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :return: :math:`\\lVert x - \\mu \\rVert^2 - r^2`, shape :math:`B`
        """
        # squeeze class dimension
        return self._center(x).squeeze(1) - self.radius.pow(2)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :param y: target labels of shape :math:`B`; samples with labels :math:`< 0` are discarded.
            If not given, all samples are assumed to be ID.
        :return: loss :math:`\\max\\{0, \\lVert x - \\mu \\rVert^2 - r^2\\}`; one entry per ID sample if the
            reduction is ``none``
        """
        if y is not None:
            y, x = drop_unknown(y, x)
        if len(x) == 0 and self.reduction == "mean":
            return x.sum() * 0.0
        # the reference soft-boundary objective is R^2 + 1/nu * mean(max(0, d - R^2)); for a fixed R,
        # R^2 is a constant and 1/nu only scales the loss, so both are omitted
        loss = DeepSVDDLoss.svdd_loss(x, self.center, radius=self.radius)
        return apply_reduction(loss, self.reduction)

    @staticmethod
    def svdd_loss(
        x: Tensor,
        center: ClassCenters,
        radius: Tensor = 0.0,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculates the loss. Treats all ID samples equally, and ignores all OOD samples.
        If no labels are given, assumes all samples are IN.

        :param x: features of shape :math:`B \\times D`
        :param center: center of sphere
        :param radius: radius of sphere
        :param y: Optional labels of shape :math:`B`.
        :return: per-sample loss of shape :math:`B`
        """
        radius = torch.as_tensor(radius)
        if y is not None:
            known = is_known(y)
        else:
            known = torch.ones(size=(x.shape[0],)).bool()

        loss = torch.zeros(size=(x.shape[0],)).to(x.device)

        if known.any():
            loss[known] = (center(x[known]).squeeze(1) - radius.pow(2)).relu()

        return loss


class DeepSADLoss(torch.nn.Module):
    """
    Deep Semi-supervised Anomaly Detection (Deep SAD), the semi-supervised generalization of
    Deep Support Vector Data Description, which also uses labeled outliers.
    It places a center :math:`\\mu` in the output space of the model and pulls ID samples towards this center in order
    to learn the common factors of intra class variance.

    The distance of a representation to this center can be used as outlier score for the corresponding input.
    Samples with targets :math:`\\geq 0` are pulled towards the center, and labeled outliers (targets :math:`< 0`)
    are pushed away from it by minimizing the inverse of their squared distance. The per-sample loss is

    .. math::
        \\begin{cases}
            \\lVert f(x) - \\mu \\rVert_2^2 & \\text{if } y \\geq 0 \\\\
            \\eta \\, (\\lVert f(x) - \\mu \\rVert_2^2 + \\epsilon)^{-1} & \\text{if } y < 0
        \\end{cases}

    .. warning:: The paper also distinguishes labeled from unlabeled normal samples and weights all
        labeled samples with :math:`\\eta`. Here, all samples with targets :math:`\\geq 0` are treated
        as unlabeled normal data, so :math:`\\eta` weights only the labeled outliers.

    In the original paper, the center is initialized with the mean of :math:`f(x)` over the dataset before
    training; pass it as ``center``.

    .. note:: The center :math:`\\mu` is stored in this loss, so move it to the device of the model with
        ``.to(device)``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.loss import DeepSADLoss

        # no bias terms, which would allow mapping every input to the center
        encoder = torch.nn.Linear(10, 2, bias=False)
        x, y = torch.randn(32, 10), torch.zeros(32, dtype=torch.long)
        y[:4] = -1  # labeled outliers
        # initialize the center with the mean of the initial outputs
        with torch.no_grad():
            center = encoder(x).mean(dim=0)
        criterion = DeepSADLoss(n_dim=2, center=center)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)

        loss = criterion(encoder(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = criterion.distance(encoder(x)).detach()  # outlier scores
    """

    info = LossInfo(
        paper=Paper(
            title="Deep Semi-Supervised Anomaly Detection",
            venue="ICLR",
            year=2020,
            url="https://arxiv.org/abs/1906.02694",
            code="https://github.com/lukasruff/Deep-SAD-PyTorch",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.FEATURES},
        supervised=True,
    )

    def __init__(
        self,
        n_dim: int,
        eta: float = 1.0,
        eps: float = 1e-6,
        reduction: Optional[str] = "mean",
        center: Optional[Tensor] = None,
    ):
        """
        :param n_dim: dimensionality :math:`n` of the output space
        :param eta: weight :math:`\\eta` of the labeled outliers
        :param eps: added to the squared distance of labeled outliers before inverting it, so that
            outliers at the center give a finite loss
        :param reduction: reduction method to apply, one of ``mean``, ``sum`` or ``none``
        :param center: position of the center :math:`\\mu \\in \\mathbb{R}^n` where :math:`n` is the
            dimensionality of the output space
        """
        super(DeepSADLoss, self).__init__()
        self._center = ClassCenters(1, n_dim, fixed=True)
        if center is not None:
            assert center.shape == (n_dim,)
            self._center.params.data = center.reshape(1, n_dim)
        self.eta = eta
        self.eps = eps
        self.reduction = reduction

    @property
    def center(self) -> ClassCenters:
        """
        The center :math:`\\mu`
        """
        return self._center

    def distance(self, x: Tensor) -> Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :return: :math:`\\lVert x - \\mu \\rVert^2`, shape :math:`B`
        """
        return self._center(x).squeeze(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :param y: target labels of shape :math:`B`; labels :math:`< 0` are labeled outliers
        :return: the loss; of shape :math:`B` if the reduction is ``none``
        """
        known = is_known(y)
        d = self.distance(x)
        loss = torch.zeros_like(d)
        loss[known] = d[known]
        # eps as in the reference implementation
        loss[~known] = self.eta / (d[~known] + self.eps)
        return apply_reduction(loss, self.reduction)
