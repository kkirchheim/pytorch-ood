"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.ViM
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..api import FeaturesDetector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class ViM(FeaturesDetector):
    """
    Implements Virtual Logit Matching (ViM) from the paper *ViM: Out-Of-Distribution with Virtual-logit Matching*.

    :see Paper:
        `ArXiv <https://arxiv.org/abs/2203.10807>`__
    :see Implementation:
        `GitHub <https://github.com/haoqiwang/vim/>`__

    .. note::
        Requires PyTorch ≥ 1.9 (``torch.linalg``).
    """

    requires_fit = True

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`: the principal-subspace
    #: dimension ``d``. Candidates exceeding a model's feature dimension are clamped during
    #: fitting, so this single space is safe across architectures of different width.
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
        :param w: weights :math:`W` of the last layer of the network
        :param b: biases :math:`b` of the last layer of the network
        """
        super(ViM, self).__init__()
        self.encoder = encoder
        self.d = d
        w = w.detach().cpu().float()
        b = b.detach().cpu().float()
        self.w = w  # (C, D)
        self.b = b  # (C,)
        self.u = -(torch.linalg.pinv(w) @ b)  # (D,)  new origin
        self.principal_subspace: Optional[Tensor] = None
        self.alpha: Optional[float] = None  #: the computed :math:`\alpha` value

    def _get_logits(self, features: Tensor) -> Tensor:
        """
        Calculates logits from features.
        """
        return features @ self.w.T + self.b

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model input, will be passed through neural network
        """
        if self.encoder is None:
            raise ModelNotSetException

        if self.principal_subspace is None or self.alpha is None:
            raise RequiresFittingException()

        with torch.no_grad():
            features = self.encoder(x)

        return self.predict_features(features)

    def __repr__(self):
        return f"ViM(d={self.d})"

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extracts features and logits, computes principle subspace and alpha. Ignores OOD samples.

        :param data_loader: dataset to fit on
        """
        if self.encoder is None:
            raise ModelNotSetException

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        features, labels = extract_features(data_loader, self.encoder, device)
        return self.fit_features(features, labels)

    @torch.no_grad()
    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features as given by the model
        """
        device = self.w.device
        x = x.detach().to(device).float()
        logits = self._get_logits(x)  # (N, C)

        # Project centered features onto the null subspace and take L2 norm
        x_p_t = (x - self.u) @ self.principal_subspace  # (N, D-d)
        vlogit = x_p_t.norm(dim=-1) * self.alpha  # (N,)

        # Clip for numerical stability: float32 easily overflows in logsumexp
        energy = torch.logsumexp(logits.clamp(-100, 100), dim=-1)  # (N,)

        score = -vlogit + energy
        return -score

    def fit_features(self: Self, features: Tensor, labels: Tensor) -> Self:
        """
        Extracts features and logits, computes principle subspace and alpha. Ignores OOD samples.

        :param features: features
        :param labels: class labels
        :return:
        """
        device = self.device or self.w.device
        features = features.detach().to(device).float()

        if features.shape[1] < self.d:
            n = features.shape[1] // 2
            log.warning(f"{features.shape[1]=} is smaller than {self.d=}. Will be adjusted to {n}")
            self.d = n

        logits = self._get_logits(features)  # (N, C)

        log.info("Computing principal space ...")
        X = features - self.u  # (N, D)  centered features

        # Empirical covariance (assume_centered=True → MLE: divide by n)
        cov = (X.T @ X) / X.shape[0]  # (D, D)

        # Eigendecomposition of the symmetric covariance matrix.
        # torch.linalg.eigh returns eigenvalues in ascending order with
        # corresponding eigenvectors as columns.
        eig_vals, eig_vecs = torch.linalg.eigh(cov)  # vals: (D,), vecs: (D, D)

        # Select the null subspace: the (D - d) eigenvectors that correspond
        # to the *smallest* eigenvalues (i.e. directions least explained by the
        # training data).  With ascending eigh output these are the first columns.
        k = eig_vecs.shape[1] - self.d
        self.principal_subspace = eig_vecs[:, :k].contiguous()  # (D, D-d)

        log.info("Computing alpha ...")
        x_p_t = X @ self.principal_subspace  # (N, D-d)
        vlogits = x_p_t.norm(dim=-1)  # (N,)
        self.alpha = (logits.max(dim=-1).values.mean() / vlogits.mean()).item()
        log.info(f"{self.alpha=:.4f}")
        return self
