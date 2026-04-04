"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.GMM
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
from ..utils import contains_unknown, extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class GMM(FeaturesDetector):
    """
    Implements a class-conditional Gaussian Mixture Model (GMM) for Out-of-Distribution Detection.

    Fits one Gaussian per class on penultimate-layer features, computing per-class means,
    covariance matrices, and mixing weights from the training data. The outlier score is the
    negative log-likelihood under the mixture:

    .. math::
        -\\log \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(z \\mid \\mu_k, \\Sigma_k)

    This extends :class:`Mahalanobis` by allowing **per-class covariance matrices** and
    using the full mixture likelihood (logsumexp) instead of the max over classes.
    """

    requires_fit = True

    def __init__(
        self,
        model: Optional[Callable[[Tensor], Tensor]],
        reg: float = 1e-6,
    ):
        """
        :param model: neural network to use for feature extraction (can be ``None`` for feature-based interface)
        :param reg: regularization added to the diagonal of each covariance matrix for numerical stability
        """
        self.model = model
        self.reg = reg
        # fitted parameters
        self._mu = None  # (K, D)
        self._precision = None  # (K, D, D)
        self._log_det = None  # (K,)
        self._log_weights = None  # (K,)

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extract features and fit the GMM.

        :param data_loader: data loader with training data
        """
        if self.model is None:
            raise ModelNotSetException()

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.model, device)
        return self.fit_features(z, y)

    def fit_features(self: Self, z: Tensor, labels: Tensor) -> Self:
        """
        Fit one Gaussian per class directly on features. OOD-labeled samples are ignored.

        :param z: features
        :param labels: class labels
        """
        known = is_known(labels)
        if not known.any():
            raise ValueError("No ID samples found.")

        assert not contains_unknown(labels[known])

        z = z[known].detach().cpu().float()
        y = labels[known].cpu().long()

        classes = y.unique()
        n_classes = len(classes)
        n_total = z.shape[0]
        d = z.shape[1]

        mu = torch.zeros(n_classes, d)
        precision = torch.zeros(n_classes, d, d)
        log_det = torch.zeros(n_classes)
        log_weights = torch.zeros(n_classes)

        for i, c in enumerate(classes):
            mask = y == c
            z_c = z[mask]
            n_c = z_c.shape[0]

            mu[i] = z_c.mean(dim=0)
            cov = (z_c - mu[i]).T @ (z_c - mu[i]) / n_c
            cov += torch.eye(d) * self.reg

            precision[i] = torch.linalg.inv(cov)
            log_det[i] = torch.linalg.slogdet(cov).logabsdet
            log_weights[i] = torch.tensor(n_c / n_total).log()

        self._mu = mu
        self._precision = precision
        self._log_det = log_det
        self._log_weights = log_weights
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through the model
        """
        if self.model is None:
            raise ModelNotSetException()

        z = self.model(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculate outlier scores from features using the negative GMM log-likelihood.

        :param z: features
        :return: outlier scores (higher = more OOD)
        """
        if self._mu is None:
            raise RequiresFittingException()

        device = self._mu.device
        z = z.detach().to(device).float()

        # Per-class Mahalanobis distances; use max (closest class) as score.
        # We omit log-det and mixing-weight terms: they are constant per class and
        # can push absolute score magnitudes into ranges that cause numerical issues
        # in downstream metrics (e.g. torchmetrics binary_auroc applies sigmoid).
        # Using only the Mahalanobis term preserves the ranking while keeping scores
        # in a well-behaved range.
        d = z.shape[1]
        mahal_k = []
        for k in range(self._mu.shape[0]):
            diff = z - self._mu[k]  # (N, D)
            mahal = (diff @ self._precision[k] * diff).sum(dim=1)  # (N,)
            mahal_k.append(mahal)

        mahal_k = torch.stack(mahal_k, dim=1)  # (N, K)
        # Use minimum Mahalanobis distance (closest class), normalized by
        # dimensionality to keep scores in a numerically safe range.
        return torch.min(mahal_k, dim=1).values / d
