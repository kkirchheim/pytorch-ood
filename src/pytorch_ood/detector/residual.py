"""

..  autoclass:: pytorch_ood.detector.Residual
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
import warnings
from typing import Callable, Optional

import torch
from torch import Tensor
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils.utils import _gram_matrix, _linalg_dtype

log = logging.getLogger(__name__)


class Residual(FeaturesDetector):
    """
    Implements the Residual detector from the paper *ViM: Out-Of-Distribution with Virtual-logit Matching*.

    The (centered) features are decomposed into a principal subspace of dimension :math:`d` and its
    orthogonal residual. The outlier score is the norm of the residual:

    .. math::
        \\lVert x^{P\\perp} \\rVert

    where :math:`x^{P\\perp}` is the residual of the features :math:`x` after centering at the origin
    :math:`u = -W^{+} b` and projecting out the principal subspace, and :math:`W` and :math:`b` are the
    weights and biases of the last layer.

    The residual subspace is computed in double precision (single precision on MPS, which does not
    support double precision).

    This detector serves as the base class of :class:`ViM <pytorch_ood.detector.ViM>`, which
    additionally scales the residual norm by a factor :math:`\\alpha` and subtracts the energy of
    the logits.

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = Residual(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
        detector.fit(train_loader)
        scores = detector(images)
    """

    info = DetectorInfo(
        paper=Paper(
            title="ViM: Out-Of-Distribution with Virtual-logit Matching",
            venue="CVPR",
            year=2022,
            url="https://arxiv.org/abs/2203.10807",
            code="https://github.com/haoqiwang/vim/",
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`: the principal-subspace
    #: dimension ``d``. If ``d`` is not smaller than the feature dimension :math:`D`, fitting uses
    #: :math:`D // 2` instead.
    # This lets a single search space work across architectures of different width.
    hyperparameter_space = {"d": [16, 32, 64, 128, 256, 512]}

    def __init__(
        self,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]],
        d: int,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        :param encoder: feature encoder. Can be
            ``None`` when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param d: dimensionality of the principal subspace
        :param w: weights :math:`W` of the last layer of the network, of shape :math:`C \\times D`
        :param b: biases :math:`b` of the last layer of the network, of shape :math:`C`
        """
        super(Residual, self).__init__()
        self.encoder = encoder
        self.d = d
        w = w.detach().float()
        b = b.detach().float()
        self.w = w  # (C, D)
        self.b = b  # (C,)
        dtype = _linalg_dtype(w.device)
        self.u = -(torch.linalg.pinv(w.to(dtype)) @ b.to(dtype)).float()  # (D,)  new origin
        #: orthonormal basis of the residual subspace, of shape :math:`D \\times (D - d)`
        self.residual_subspace: Optional[Tensor] = None

    def _residual_norm(self, x: Tensor) -> Tensor:
        """
        Projects centered features onto the residual subspace and returns the L2 norm.

        :param x: features of shape :math:`N \\times D`
        :return: residual norms of shape :math:`N`
        """
        x_p_t = (x - self.u) @ self.residual_subspace  # (N, D-d)
        return x_p_t.norm(dim=-1)  # (N,)

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d})"

    @property
    def principal_subspace(self) -> Optional[Tensor]:
        """
        Deprecated alias of :attr:`residual_subspace <pytorch_ood.detector.Residual.residual_subspace>`.
        """
        warnings.warn(
            "'principal_subspace' is deprecated, use 'residual_subspace' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.residual_subspace

    @torch.no_grad()
    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        """
        if self.residual_subspace is None:
            raise RequiresFittingException()

        x = x.detach().to(self.w.device).float()
        return self._residual_norm(x)

    def fit_features(self, features: Tensor, labels: Tensor) -> Self:
        """
        Computes the principal subspace from pre-extracted features. Ignores OOD samples.

        :param features: features of shape :math:`N \\times D`
        :param labels: class labels of shape :math:`N`
        :return: self
        """
        device = self.device or self.w.device
        features = features.detach().to(device).float()

        d = self.d
        if d >= features.shape[1]:  # d == D would leave an empty residual subspace
            d = features.shape[1] // 2
            log.warning(f"{features.shape[1]=} is not larger than {self.d=}. Using d={d}")

        log.info("Computing principal space ...")
        # Empirical covariance (assume_centered=True → MLE: divide by n), in float64: centered at u, not
        # at the mean, its leading eigenvalue is large and float32 eigenvectors of the small ones inaccurate
        u = self.u.to(_linalg_dtype(features.device))
        cov = _gram_matrix(features, transform=lambda c: c - u) / features.shape[0]  # (D, D)

        # Eigendecomposition of the symmetric covariance matrix.
        # torch.linalg.eigh returns eigenvalues in ascending order with
        # corresponding eigenvectors as columns.
        eig_vals, eig_vecs = torch.linalg.eigh(cov)  # vals: (D,), vecs: (D, D)

        # Select the null subspace: the (D - d) eigenvectors that correspond
        # to the *smallest* eigenvalues (i.e. directions least explained by the
        # training data).  With ascending eigh output these are the first columns.
        k = eig_vecs.shape[1] - d
        self.residual_subspace = eig_vecs[:, :k].float().contiguous()  # (D, D-d)
        return self
