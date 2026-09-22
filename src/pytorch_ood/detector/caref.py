"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.CARef
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
from ..utils import extract_features, is_known

Self = TypeVar("Self")
log = logging.getLogger(__name__)


class CARef(FeaturesDetector):
    """
    Implements CARef from the paper
    *CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature
    Leveraging*.

    CARef measures how far the penultimate-layer features of an input deviate from the average
    features of the class the model assigns to it. The deviation is an :math:`l_1` distance,
    normalized by the :math:`l_1` norm of the features themselves, which makes it comparable
    across inputs with different activation magnitude.

    During fitting, the average feature vector is estimated for each class :math:`k`

    .. math::
        \\mu^{k} = \\frac{1}{n_k} \\sum_{x \\in \\mathcal{D}_{train}}
        \\mathbb{1}(\\hat{y}(x) = k) \\, f(x)

    where :math:`f(x)` are the features and :math:`\\hat{y}(x)` is the class predicted by the
    model. Following the paper and the reference implementation, the centroids are conditioned
    on the **predicted** label rather than on the ground-truth label; labels passed to
    :meth:`fit_features` are only used to discard OOD samples. The outlier score for an input
    with predicted class :math:`\\hat{y}` is the relative error

    .. math::
        s(x) = \\frac{\\lVert f(x) - \\mu^{\\hat{y}} \\rVert_1}{\\lVert f(x) \\rVert_1}

    This method is hyperparameter-free. Only pooled feature vectors of shape :math:`(N, D)`
    are supported, so it cannot be used for anomaly segmentation; grid-like inputs of shape
    :math:`(N, C, H, W)` are rejected with a ``ValueError``. Features are cast to ``float32``,
    and the normalizing :math:`\\lVert f(x) \\rVert_1` is clamped to a small positive value, so
    that an all-zero feature vector yields a large finite score instead of ``NaN``; the
    reference implementation does neither.

    .. note::
        The original publication defines :math:`\\text{Score}_{CARef} = -s(x)`, so that
        in-distribution inputs obtain the larger value. This implementation returns
        :math:`s(x)` itself, following this library's convention that larger scores indicate
        outliers.

    .. note::
        Classes that the model never predicts on the fitting data have no centroid. The
        reference implementation leaves those entries empty, which turns into ``NaN`` scores;
        this implementation falls back to the global mean of the fitting features and emits a
        warning. On a small or skewed fitting set this can affect many classes, so check the
        warning rather than ignoring it -- the affected inputs are then scored against a
        global mean rather than a class mean, which is not what the method intends.

    Example Code:

    .. code :: python

        model = WideResNet().eval()
        detector = CARef(encoder=model.features, head=model.fc)
        detector.fit(train_loader)
        scores = detector(images)

    :see Paper:
        `ArXiv <https://arxiv.org/abs/2503.00325>`__

    :see Implementation:
        `GitHub <https://github.com/LingAndZero/CADRef>`__

    """

    requires_fit = True

    #: features with a smaller :math:`l_1` norm than this are treated as having this norm
    _NORM_EPS = 1e-12

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        head: Callable[[Tensor], Tensor],
    ) -> None:
        """
        :param encoder: model mapping inputs to penultimate-layer features. Can be ``None``
            when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param head: callable mapping features to logits, used to obtain the predicted class.
            Required, both for fitting and for scoring.
        """
        super(CARef, self).__init__()

        if head is None:
            raise ValueError("head must not be None")

        self.encoder = encoder
        self.head = head
        self.train_means: Optional[Tensor] = None  #: average feature per class, shape (C, D)

    def __repr__(self):
        return "CARef()"

    @staticmethod
    def _as_features(z: Tensor) -> Tensor:
        """
        Validate and normalize a feature tensor to shape ``(N, D)`` of floating dtype.
        """
        if z.ndim != 2:
            raise ValueError(
                f"Expected pooled features with shape (N, D), got {tuple(z.shape)}. "
                "This detector does not support grid-like inputs."
            )

        return z.detach().float()

    @torch.no_grad()
    def _prepare_fit_features(self, z: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Validate the fitting features and discard OOD samples.

        The features are deliberately left on their original device. Fitting moves them to the
        detector's device chunk by chunk, so that a training set too large for device memory
        can still be used.
        """
        z = self._as_features(z)

        if z.shape[0] == 0:
            raise ValueError("No samples to fit on")

        if y is not None:
            known = is_known(y).to(z.device)
            if not known.any():
                raise ValueError("No ID samples")
            z = z[known]

        return z

    @torch.no_grad()
    def _fit_class_means(self, z: Tensor, batch_size: int) -> Tensor:
        """
        Average feature vector per predicted class, shape ``(C, D)``.

        Accumulated chunk-wise, so peak device memory is bounded by ``batch_size`` and by the
        ``(C, D)`` accumulator rather than by the size of the fitting set.
        """
        device = self.device or z.device
        sums: Optional[Tensor] = None
        counts: Optional[Tensor] = None

        for start in range(0, z.shape[0], batch_size):
            z_batch = z[start : start + batch_size].to(device)
            logits = self.head(z_batch)

            if sums is None:
                n_classes = logits.shape[1]
                sums = torch.zeros(n_classes, z_batch.shape[1], dtype=z_batch.dtype, device=device)
                counts = torch.zeros(n_classes, dtype=z_batch.dtype, device=device)

            y_hat = logits.argmax(dim=1)
            sums.index_add_(0, y_hat, z_batch)
            counts.index_add_(0, y_hat, torch.ones_like(y_hat, dtype=z_batch.dtype))

        means = sums / counts.clamp(min=1.0).unsqueeze(1)

        empty = counts == 0
        if empty.any():
            log.warning(
                f"{int(empty.sum())} of {counts.shape[0]} classes were never predicted on the "
                "fitting data. Their centroids fall back to the global feature mean."
            )
            # every sample is assigned to exactly one class, so the column sums are the totals
            means[empty] = sums.sum(dim=0) / counts.sum()

        return means

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extract features and estimate the class-aware average features.

        :param data_loader: dataset to fit on, usually the training dataset
        """
        if self.encoder is None:
            raise ModelNotSetException

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device=device)
        return self.fit_features(z, y)

    @torch.no_grad()
    def fit_features(
        self: Self, z: Tensor, y: Optional[Tensor] = None, batch_size: int = 4096
    ) -> Self:
        """
        Estimate the class-aware average features. Ignores OOD samples.

        :param z: features of the training data, shape ``(N, D)``. May live on a different
            device than the detector; chunks are moved as needed.
        :param y: corresponding class labels. When given, OOD samples are discarded. The
            labels are not used to form the centroids, see the class documentation.
        :param batch_size: chunk size used while passing features through the head
        """
        z = self._prepare_fit_features(z, y)
        self.train_means = self._fit_class_means(z, batch_size)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate outlier scores for inputs, which will be passed through the encoder.

        :param x: model input, will be passed through the encoder
        """
        if self.encoder is None:
            raise ModelNotSetException

        if self.train_means is None:
            raise RequiresFittingException()

        with torch.no_grad():
            z = self.encoder(x)

        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculate outlier scores based on features.

        :param z: features as given by the model, shape ``(N, D)``
        :return: outlier scores, higher means more outlier
        """
        if self.train_means is None:
            raise RequiresFittingException()

        device = self.device or z.device
        z = self._as_features(z).to(device)

        centroids = self.train_means[self.head(z).argmax(dim=1)]
        return (z - centroids).norm(p=1, dim=1) / z.norm(p=1, dim=1).clamp(min=self._NORM_EPS)
