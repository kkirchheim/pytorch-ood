"""

..  autoclass:: pytorch_ood.detector.RMD
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

from pytorch_ood.detector.mahalanobis import Mahalanobis

from ..api import DetectorInfo, ModelNotSetException, Paper, RequiresFittingException, Task
from ..utils import (
    extract_features,
    is_known,
)
from ..utils.utils import _fit_gaussian, _squared_mahalanobis

log = logging.getLogger(__name__)


class RMD(Mahalanobis):
    """
    Implements the Relative Mahalanobis Distance (RMD) from the paper
    *A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection*.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.

    Additionally, it fits a background gaussian with mean :math:`\\mu_0` and covariance matrix
    :math:`\\Sigma_0` to all of the features and calculates outlier scores as

    .. math :: \\min_k \\lbrace d_k(f(x)) - d_0(f(x)) \\rbrace

    where :math:`f(x)` are the features of :math:`x`, :math:`d_k(z) = (z - \\mu_k)^\\top \\Sigma^{-1} (z - \\mu_k)` is
    the squared Mahalanobis distance to class :math:`k` and :math:`d_0` is the
    squared Mahalanobis distance under the background gaussian.

    Both covariances are estimated as in :class:`~pytorch_ood.detector.Mahalanobis`, in double
    precision (single precision on MPS, which does not support double precision).
    """

    info = DetectorInfo(
        paper=Paper(
            title="A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection",
            venue="arXiv",
            year=2021,
            url="https://arxiv.org/pdf/2106.09022.pdf",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
    ):
        """
        :param encoder: feature extractor that maps inputs to features of shape :math:`B \\times D`.
            Can be ``None`` when using ``fit_features(...)`` and ``predict_features(...)``
            directly.
        """
        super(RMD, self).__init__(encoder=encoder)

        self.background_mu = None
        self.background_cov = None
        self.background_precision = None
        self._background_whitening: Tensor = None
        self._background_whitened_mu: Tensor = None

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Fit parameters of the multivariate Gaussians for the given loader.
        Ignores OOD Inputs.

        :param data_loader: data loader with training data
        :return: self
        :raises ValueError: if the regularized covariance is singular
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device=device)
        return self.fit_features(z, y)

    def fit_features(self, z: Tensor, y: Tensor) -> Self:
        """
        Fit parameters of the multivariate Gaussians. Ignores OOD inputs.

        :param z: features of shape :math:`N \\times D`
        :param y: class labels of shape :math:`N`
        :return: self
        :raises ValueError: if the regularized covariance is singular
        """
        device = self.device or z.device

        # y = y.to(device)
        known = is_known(y)

        super(RMD, self).fit_features(z, y)

        z_known = z[known].to(device)

        log.debug("Fitting background gaussian.")
        # one class covering all known samples
        background_y = torch.zeros(len(z_known), dtype=torch.long, device=device)
        background = _fit_gaussian(
            z_known, background_y, n_classes=1, eps=self.eps, use_pinv=self.use_pinv
        )
        self.background_mu = background.mu[0]
        self.background_cov = background.cov
        self.background_precision = background.precision
        self._background_whitening = background.whitening
        self._background_whitened_mu = background.whitened_mu
        return self

    def _is_fitted(self) -> bool:
        return super()._is_fitted() and getattr(self, "_background_whitening", None) is not None

    def _background_score(self, z: Tensor) -> Tensor:
        return _squared_mahalanobis(
            z, self._background_whitening, self._background_whitened_mu
        ).squeeze(1)

    def _calc_gaussian_scores(self, z: Tensor) -> Tensor:
        """
        This is written a bit differently compared to the mahalanobis paper
        """
        # mean over feature maps
        features = z.view(z.size(0), z.size(1), -1)
        features = torch.mean(features, 2)
        return _squared_mahalanobis(features, self._whitening, self._whitened_mu)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculates the relative Mahalanobis distance directly on features.

        :param z: features of shape :math:`B \\times D` (or feature maps of shape
            :math:`B \\times C \\times H \\times W`, which are spatially averaged)
        :return: relative Mahalanobis scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        """

        if not self._is_fitted():
            raise RequiresFittingException

        z = z.to(self.mu.dtype)
        md_k = self._calc_gaussian_scores(z)
        md_0 = self._background_score(z)

        score = torch.min(md_k - md_0.view(-1, 1), dim=1).values
        return score

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input batch, will be passed through the encoder
        :return: relative Mahalanobis scores of shape :math:`B`
        :raises ModelNotSetException: if ``encoder`` is ``None``
        :raises RequiresFittingException: if the detector was not fitted
        """
        if self.encoder is None:
            raise ModelNotSetException

        features = self.encoder(x)
        return self.predict_features(features)
