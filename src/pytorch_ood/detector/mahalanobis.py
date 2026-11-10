"""

..  autoclass:: pytorch_ood.detector.Mahalanobis
    :members:
    :inherited-members:
    :show-inheritance:
"""

import logging
import warnings
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    GradientDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils import (
    contains_unknown,
    extract_features,
)
from ..utils.utils import _fit_gaussian, _squared_mahalanobis

log = logging.getLogger(__name__)


class Mahalanobis(FeaturesDetector):
    """
    Implements the Mahalanobis Method from the paper *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks*.

    This method calculates a class center :math:`\\mu_k` for each class :math:`k`,
    and a shared covariance matrix :math:`\\Sigma` from the data.
    The outlier score is half the squared Mahalanobis distance to the closest class center:

    .. math :: \\frac{1}{2} \\min_k (f(x) - \\mu_k)^{\\top} \\Sigma^{-1} (f(x) - \\mu_k)

    The paper uses the negated distance as a confidence score, so the sign is flipped.

    :math:`\\Sigma` is the shared within-class covariance, i.e., the centered outer products of the
    features divided by the number of samples. To ensure that it is invertible, a small value
    :math:`\\epsilon` is added to its diagonal before inversion, i.e., :math:`\\Sigma + \\epsilon I` is
    used. Alternatively, the pseudo-inverse of :math:`\\Sigma` can be used (``use_pinv``), which
    ignores directions in which the training features do not vary. This is preferable when there are
    fewer training samples than feature dimensions.

    The parameters are estimated in double precision (single precision on MPS, which does not support
    double precision) and stored in the dtype of the features.
    """

    info = DetectorInfo(
        paper=Paper(
            title="A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks",
            venue="NeurIPS",
            year=2018,
            url="https://arxiv.org/abs/1807.03888",
            code="https://github.com/pokaxpoka/deep_Mahalanobis_detector",
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        # tiny, since the covariance is divided by N and inverted in float64
        eps: float = 1e-10,
        use_pinv: bool = False,
    ):
        """
        :param encoder: feature encoder that maps inputs to features of shape :math:`B \\times D`. Can be
            ``None`` when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param eps: regularization constant :math:`\\epsilon` added to the diagonal of the covariance matrix
            before inversion, so that the matrix is invertible. Set to ``0`` to disable regularization.
            Not used if ``use_pinv`` is ``True``.
        :param use_pinv: if ``True``, the precision matrix is the Moore-Penrose pseudo-inverse
            (``torch.linalg.pinv``) of the unregularized covariance matrix instead of the inverse
            (``torch.linalg.inv``) of the regularized one. Directions without variance in the training
            features then do not contribute to the distance.
        """
        super(Mahalanobis, self).__init__()
        self.encoder = encoder
        self.eps = eps
        self.use_pinv = use_pinv
        self.mu: Tensor = None  #: Centers
        self.cov: Tensor = None  #: Covariance Matrix
        self.precision: Tensor = None  #: Precision Matrix
        # factor L of the precision (P = L L^T) and mu @ L, used for scoring
        self._whitening: Tensor = None
        self._whitened_mu: Tensor = None

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Fit parameters of the multivariate Gaussian.

        :param data_loader: dataset to fit on. Should only contain in-distribution samples, with
            contiguous class labels :math:`0, \\ldots, C-1`, all of which have to occur.
        :return: the fitted detector
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device)
        return self.fit_features(z, y)

    def fit_features(self, z: Tensor, y: Tensor) -> Self:
        """
        Fit parameters of the multivariate Gaussian.

        :param z: features of shape :math:`N \\times D`. Should only contain in-distribution samples.
        :param y: class labels of shape :math:`N`, contiguous :math:`0, \\ldots, C-1`; all classes have to
            occur. OOD samples (labels below zero) are not supported and raise an ``AssertionError``.
        :return: the fitted detector
        """
        device = self.device or z.device
        z = z.detach().to(device)
        y = y.to(device)

        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        # as the reference implementation (sklearn EmpiricalCovariance): divided by N, float64
        gaussian = _fit_gaussian(
            z, y, n_classes=len(classes), eps=self.eps, use_pinv=self.use_pinv
        )
        self.mu, self.cov, self.precision = gaussian.mu, gaussian.cov, gaussian.precision
        self._whitening, self._whitened_mu = gaussian.whitening, gaussian.whitened_mu
        return self

    def _calc_gaussian_scores(self, z: Tensor) -> Tensor:
        """ """
        features = z.view(z.size(0), z.size(1), -1)
        features = torch.mean(features, 2).to(self.mu.dtype)
        # the factor -0.5 (Gaussian log-density term) is taken from the reference implementation
        return -0.5 * _squared_mahalanobis(features, self._whitening, self._whitened_mu)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculates mahalanobis distance directly on features.
        The input preprocessing of :class:`~pytorch_ood.detector.MahalanobisODIN` is not applied.

        :param z: features, as given by the model, of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        """
        if self.mu is None:
            raise RequiresFittingException

        md_k = self._calc_gaussian_scores(z)
        score = -torch.max(md_k, dim=1).values
        return score

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through ``encoder``
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if ``encoder`` is ``None``
        :raises RequiresFittingException: if the detector was not fitted
        """
        if self.encoder is None:
            raise ModelNotSetException

        features = self.encoder(x)
        return self.predict_features(features)

    @property
    def n_classes(self):
        """
        Number of classes the model is fitted for
        """
        if self.mu is None:
            raise RequiresFittingException

        return self.mu.shape[0]


class MahalanobisODIN(GradientDetector):
    """
    Mahalanobis distance detector with input preprocessing, as proposed in *A Simple Unified
    Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*: the input is
    perturbed in the style of :class:`ODIN <pytorch_ood.detector.ODIN>` before the Mahalanobis
    distance is computed.

    The perturbation is gradient-guided (FGSM-style), so prediction requires gradient
    computation:

    .. math:: x' = x - \\varepsilon \\cdot \\text{sign}(\\nabla_x L(x))

    where :math:`L` is half the squared Mahalanobis distance of the features of :math:`x` to the closest class
    center. The outlier score is the score of
    :class:`Mahalanobis <pytorch_ood.detector.Mahalanobis>` for :math:`x'`.
    """

    info = DetectorInfo(
        paper=Paper(
            title="A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks",
            venue="NeurIPS",
            year=2018,
            url="https://arxiv.org/abs/1807.03888",
            code="https://github.com/pokaxpoka/deep_Mahalanobis_detector",
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    def __init__(
        self,
        encoder: Callable[[Tensor], Tensor],
        eps: float = 0.002,
        norm_std: Optional[List[float]] = None,
    ):
        """
        :param encoder: feature encoder that maps inputs to features of shape :math:`B \\times D`; used to
            perturb and score the inputs in :meth:`predict <pytorch_ood.detector.MahalanobisODIN.predict>`.
            :meth:`fit_features <pytorch_ood.detector.MahalanobisODIN.fit_features>` fits on precomputed
            features.
        :param eps: step size :math:`\\varepsilon` of the gradient based input preprocessing, in units of the
            input :math:`x`
        :param norm_std: per-channel standard deviations of the input normalization, for details see
            :class:`ODIN <pytorch_ood.detector.ODIN>`
        """
        super(MahalanobisODIN, self).__init__()
        self._base = Mahalanobis(encoder)
        self.eps = eps
        self.norm_std = norm_std

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Fit the underlying Mahalanobis detector.

        :param data_loader: dataset to fit on.
        :return: the fitted detector
        """
        self._base.fit(data_loader)
        return self

    def fit_features(self, z: Tensor, y: Tensor) -> Self:
        """
        Fit the underlying Mahalanobis detector on features.

        :param z: features of shape :math:`N \\times D`
        :param y: class labels of shape :math:`N`
        :return: the fitted detector
        """
        self._base.fit_features(z, y)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Apply ODIN input perturbation and compute Mahalanobis distance.

        :param x: input tensor
        :return: outlier scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        """
        x = self._odin_preprocess(x, x.device)
        return self._base.predict(x)

    def _odin_preprocess(self, x: Tensor, dev: str):
        """
        ODIN input preprocessing guided by Mahalanobis distance.
        """
        if torch.is_inference_mode_enabled():
            warnings.warn("ODIN not compatible with inference mode. Will be deactivated.")

        with torch.inference_mode(False):
            if torch.is_inference(x):
                x = x.clone()

            with torch.enable_grad():
                x = Variable(x, requires_grad=True)
                features = self._base.encoder(x)
                features = features.view(features.shape[0], -1)
                whitening = self._base._whitening
                distances = _squared_mahalanobis(
                    features.detach(), whitening, self._base._whitened_mu
                )
                sample_pred = distances.argmin(dim=1)
                batch_sample_mean = self._base.mu.index_select(0, sample_pred)
                # half the squared distance to the closest center, ||(f - mu_k) L||^2 / 2
                whitened = (features - batch_sample_mean) @ whitening
                loss = torch.mean(0.5 * whitened.square().sum(dim=1))
                loss.backward()

                gradient = torch.sign(x.grad.data)

        if self.norm_std:
            for i, std in enumerate(self.norm_std):
                gradient.index_copy_(
                    1,
                    torch.LongTensor([i]).to(dev),
                    gradient.index_select(1, torch.LongTensor([i]).to(dev)) / std,
                )
        perturbed_x = x.data - self.eps * gradient

        return perturbed_x
