"""

..  autoclass:: pytorch_ood.detector.GMM
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
import math
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    RequiresFittingException,
    Task,
)
from ..utils import extract_features, is_known

log = logging.getLogger(__name__)


class GMM(FeaturesDetector):
    """
    Implements a class-conditional Gaussian Mixture Model (GMM) for Out-of-Distribution Detection.

    Fits one Gaussian per class on penultimate-layer features, with the mean :math:`\\mu_k`, the
    covariance matrix :math:`\\Sigma_k`, and the mixing weight :math:`\\pi_k` (the fraction of
    training samples) of each class :math:`k`. The outlier score is the negative log-likelihood of
    the features :math:`z` under the mixture:

    .. math::
        -\\log \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(z \\mid \\mu_k, \\Sigma_k)

    This extends :class:`~pytorch_ood.detector.Mahalanobis` by allowing **per-class covariance matrices** and
    using the full mixture likelihood (logsumexp) instead of the max over classes.
    """

    # a classical baseline (class-conditional Gaussians); no single paper introduced it
    info = DetectorInfo(tasks={Task.CLASSIFICATION}, ai_coded=True)

    requires_fit = True

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        reg: float = 1e-6,
    ):
        """
        :param encoder: feature encoder (can be ``None`` for feature-based interface)
        :param reg: regularization added to the diagonal of each covariance matrix, so that it is
            invertible even with fewer samples than feature dimensions
        """
        self.encoder = encoder
        self.reg = reg
        # fitted parameters
        self._mu = None  # (K, D)
        self._precision = None  # (K, D, D)
        self._log_det = None  # (K,)
        self._log_weights = None  # (K,)

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extract features and fit the GMM.

        :param data_loader: data loader with training data. OOD samples are ignored.
        :return: the fitted detector
        :raises ModelNotSetException: if the detector has no ``encoder``
        :raises ValueError: if the data contains no in-distribution sample, or a covariance matrix
            is not positive definite even with the regularization ``reg``
        """
        if self.encoder is None:
            raise ModelNotSetException()

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device)
        return self.fit_features(z, y)

    def fit_features(self, z: Tensor, labels: Tensor) -> Self:
        """
        Fit one Gaussian per class directly on features. OOD-labeled samples are ignored.

        :param z: features of shape :math:`N \\times D`
        :param labels: class labels of shape :math:`N`
        :return: the fitted detector
        :raises ValueError: if no in-distribution sample is present, or a covariance matrix is not
            positive definite even with the regularization ``reg``
        """
        known = is_known(labels)
        if not known.any():
            raise ValueError("No ID samples found.")

        device = self.device or z.device
        dtype = _dtype(device)
        z = z[known].detach().to(device, dtype)
        y = labels[known].to(device).long()

        classes = y.unique()
        n_total, d = z.shape
        eye = torch.eye(d, device=device, dtype=dtype)

        mu, precision, log_det, log_weights = [], [], [], []
        for c in classes:
            z_c = z[y == c]
            mu_c = z_c.mean(dim=0)
            cov = (z_c - mu_c).T @ (z_c - mu_c) / z_c.shape[0] + self.reg * eye
            chol, info = torch.linalg.cholesky_ex(cov)
            if info != 0:
                raise ValueError(
                    f"The covariance matrix of class {int(c)} is not positive definite; "
                    f"increase reg (currently {self.reg})"
                )
            mu.append(mu_c)
            precision.append(torch.cholesky_inverse(chol))
            log_det.append(2 * chol.diagonal().log().sum())
            log_weights.append(torch.tensor(z_c.shape[0] / n_total, dtype=dtype).log())

        self._mu = torch.stack(mu)  # (K, D)
        self._precision = torch.stack(precision)  # (K, D, D)
        self._log_det = torch.stack(log_det)  # (K,)
        self._log_weights = torch.stack(log_weights).to(device)  # (K,)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through the encoder
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if the detector has no ``encoder``
        :raises RequiresFittingException: if the detector was not fitted
        """
        if self.encoder is None:
            raise ModelNotSetException()

        z = self.encoder(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculate outlier scores from features using the negative GMM log-likelihood.

        :param z: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        """
        if self._mu is None:
            raise RequiresFittingException()

        z = z.detach().to(self._mu.device, self._mu.dtype)
        d = z.shape[1]
        diff = z.unsqueeze(0) - self._mu.unsqueeze(1)  # (K, B, D)
        # squared Mahalanobis distance to each class, (B, K)
        mahal = ((diff @ self._precision) * diff).sum(dim=2).T
        log_likelihood = self._log_weights - 0.5 * (
            mahal + self._log_det + d * math.log(2 * math.pi)
        )
        return -torch.logsumexp(log_likelihood, dim=1)


def _dtype(device) -> torch.dtype:
    # float64 for stable covariances and log-determinants; MPS does not support it
    return torch.float32 if torch.device(device).type == "mps" else torch.float64
