import logging

import torch
import torch.nn as nn

from ..api import LossInfo, Paper, Representation, Task
from ..model.centers import ClassCenters
from ..utils import drop_unknown

log = logging.getLogger(__name__)


class CenterLoss(nn.Module):
    """
    Generalized version of the Center Loss from the Paper
    *A Discriminative Feature Learning Approach for Deep Face Recognition*.
    For each class, this loss places a center :math:`\\mu_y` in the output space and draws representations of samples
    to their corresponding class centers, up to a radius :math:`r`.

    Calculates

    .. math::
        \\mathcal{L}(x,y) = \\max \\lbrace  d(f(x),\\mu_y) - r , 0 \\rbrace

    where :math:`d` is some measure of dissimilarity, like the squared distance.
    The mean is taken over the batch and the :math:`C` classes, which is equivalent to the formula above
    divided by :math:`C`. Samples with labels :math:`< 0` are ignored.

    With radius :math:`r=0` and the squared euclidean distance as :math:`d(\\cdot,\\cdot)`, this is equivalent to
    the original center loss, which is also referred to as the *soft-margin loss* in some publications.

    .. note:: The class centers are stored in this loss, so move it to the device of the model with
        ``.to(device)``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.loss import CenterLoss

        encoder = torch.nn.Linear(10, 2)  # maps inputs into the 2-dimensional space of the centers
        criterion = CenterLoss(n_classes=3, n_dim=2)
        # the centers are learnable, so the optimizer also updates the loss
        optimizer = torch.optim.SGD([*encoder.parameters(), *criterion.parameters()], lr=0.01)

        x, y = torch.randn(8, 10), torch.randint(0, 3, (8,))
        # forward() takes the distances to the centers of this loss
        distances = criterion.distance(encoder(x))
        loss = criterion(distances, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """

    info = LossInfo(
        paper=Paper(
            title="A Discriminative Feature Learning Approach for Deep Face Recognition",
            venue="ECCV",
            year=2016,
            url="https://ydwen.github.io/papers/WenECCV16.pdf",
            code="https://github.com/KaiyangZhou/pytorch-center-loss",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.DISTANCES},
        supervised=False,
    )

    def __init__(
        self,
        n_classes: int,
        n_dim: int,
        magnitude: float = 1.0,
        radius: float = 0.0,
        fixed: bool = False,
    ):
        """
        :param n_classes: number of classes :math:`C`
        :param n_dim: dimensionality of center space :math:`D`
        :param magnitude: scale :math:`\\lambda` of the identity initialization of the centers; only applied
            if ``n_classes == n_dim`` and ``fixed=True``, otherwise the centers are drawn from a standard normal
            distribution
        :param radius: radius :math:`r` of spheres, lower bound for distance from center that is penalized
        :param fixed: false if centers should be learnable
        """
        super(CenterLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_dim
        self.magnitude = magnitude
        self.radius = radius
        self._centers = ClassCenters(n_classes=n_classes, n_features=n_dim, fixed=fixed)
        self._init_centers()

    @property
    def centers(self) -> ClassCenters:
        """
        :return: the :math:`\\mu` for all classes
        """
        return self._centers

    def distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the squared distances of the embeddings to each center, the input of
        :meth:`forward <pytorch_ood.loss.CenterLoss.forward>`.

        :param z: embeddings of shape :math:`B \\times D`
        :return: squared distances of shape :math:`B \\times C`
        """
        return self.centers(z)

    def _init_centers(self):
        # In the published code, Wen et al. initialize centers randomly.
        # However, this might bot be optimal if the loss is used without an additional
        # inter-class-discriminability term.
        # The Class Anchor Clustering initializes the centers as scaled unit vectors.
        if self.num_classes == self.feat_dim:
            torch.nn.init.eye_(self._centers._params)
            if not self._centers._params.requires_grad:
                self._centers._params.mul_(self.magnitude)
        # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
        # different then the number of classes
        # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers.params)
            if self.magnitude != 1:
                log.warning("Not applying magnitude parameter.")

    def forward(self, distmat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss. Ignores OOD inputs.

        :param distmat: matrix of distances of each point to each center with shape :math:`B \\times C`.
        :param target: ground truth labels with shape :math:`B`; labels :math:`< 0` are ignored
        :return: scalar loss
        """
        target, distmat = drop_unknown(target, distmat)
        if len(target) == 0:
            return distmat.sum() * 0.0

        batch_size = distmat.size(0)
        classes = torch.arange(self.num_classes).long().to(distmat.device)
        target = target.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = target.eq(classes.expand(batch_size, self.num_classes))
        dist = (distmat - self.radius).relu() * mask.float()
        return dist.clamp(min=1e-12, max=1e12).mean()
