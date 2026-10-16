"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.Residual
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils import extract_features

log = logging.getLogger(__name__)


class Residual(FeaturesDetector):
    """
    Implements the Residual detector from the paper
    *ViM: Out-Of-Distribution with Virtual-logit Matching* (Wang et al., CVPR 2022).

    Projects penultimate-layer features onto the **null subspace** — the subspace
    spanned by the eigenvectors of the empirical covariance matrix that correspond
    to the **smallest** eigenvalues (i.e. low-variance directions not explained by
    the training data). The L2 norm of those projected features serves as the OOD score.

    :see Paper: `CVPR 2022 <https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-Of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf>`__
    :see Implementation: `GitHub <https://github.com/haoqiwang/vim>`__
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
        super(Residual, self).__init__()
        self.encoder = encoder
        self.d = d
        w = w.detach().float()
        b = b.detach().float()
        self.w = w  # (C, D)
        self.b = b  # (C,)
        self.u = -(torch.linalg.pinv(w) @ b)  # (D,)  new origin
        self.principal_subspace: Optional[Tensor] = None

    def _get_logits(self, features: Tensor) -> Tensor:
        """
        Calculates logits from features.
        """
        return features @ self.w.T + self.b

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model input, will be passed through neural network
        :return: outlier scores of shape :math:`B`
        """
        if self.encoder is None:
            raise ModelNotSetException

        if self.principal_subspace is None:
            raise RequiresFittingException()

        with torch.no_grad():
            features = self.encoder(x)

        return self.predict_features(features)

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d})"

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extracts features and computes the principal subspace. Ignores OOD samples.

        :param data_loader: dataset to fit on
        :return: self
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
        :param x: features of shape :math:`B \times D`
        :return: outlier scores of shape :math:`B`
        """
        device = self.w.device
        x = x.detach().to(device).float()

        # Project centered features onto the null subspace and take L2 norm
        x_p_t = (x - self.u) @ self.principal_subspace  # (N, D-d)
        vlogit = x_p_t.norm(dim=-1)  # (N,)

        # Residual score is precisely the projection norm (vlogit).
        # Higher score = more likely OOD.
        return vlogit

    def fit_features(self, features: Tensor, labels: Tensor) -> Self:
        """
        Computes the principal subspace from pre-extracted features. Ignores OOD samples.

        :param features: features of shape :math:`N \times D`
        :param labels: class labels of shape :math:`N`
        :return: self
        """
        device = self.device or self.w.device
        features = features.detach().to(device).float()

        if features.shape[1] < self.d:
            n = features.shape[1] // 2
            log.warning(f"{features.shape[1]=} is smaller than {self.d=}. Will be adjusted to {n}")
            self.d = n

        log.info("Computing principal space ...")
        X = features - self.u  # (N, D)  centered features

        # Empirical covariance (assume_centered=True → MLE: divide by n)
        cov = (X.T @ X) / X.shape[0]  # (D, D)

        # Eigendecomposition of the symmetric covariance matrix.
        eig_vals, eig_vecs = torch.linalg.eigh(cov)  # vals: (D,), vecs: (D, D)

        k = eig_vecs.shape[1] - self.d
        self.principal_subspace = eig_vecs[:, :k].contiguous()  # (D, D-d)

        return self
