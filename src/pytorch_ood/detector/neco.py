"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.NECO
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

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class NECO(FeaturesDetector):
    """
    Implements NECO from the paper
    *NECO: NEural Collapse Based Out-of-distribution detection*.

    Neural Collapse predicts that, late in training, the penultimate-layer representations of
    in-distribution data converge towards a Simplex Equiangular Tight Frame (ETF) that spans only
    a low-dimensional subspace of the feature space, while out-of-distribution representations
    collapse towards the origin of that subspace. NECO exploits this by measuring how much of a
    representation's energy lies inside the ETF subspace, which is estimated as the
    :math:`d`-dimensional principal subspace of the in-distribution training features.

    Let :math:`h(x) \\in \\mathbb{R}^{D}` be the penultimate-layer representation,
    :math:`\\tilde{h}(x)` its (optionally standardized) version, :math:`\\mu` the mean of the
    training representations and :math:`P_d \\in \\mathbb{R}^{D \\times d}` the matrix holding the
    :math:`d` eigenvectors of the training covariance with the largest eigenvalues. The score is

    .. math::
        \\mathrm{NECO}(x) = \\frac{\\lVert P_d^{\\top}(\\tilde{h}(x) - \\mu) \\rVert_2}
        {\\lVert \\tilde{h}(x) \\rVert_2} \\cdot \\max_i f_i(x)

    where the multiplication with the maximum logit :math:`\\max_i f_i(x)` calibrates the ratio
    for transformer backbones. It is enabled by default; for convolutional backbones, pass
    ``use_max_logit=False``. The ratio is large for in-distribution inputs and close to zero for
    outliers, so this implementation returns the **negated** score to follow this library's
    convention that larger values indicate outliers.

    Following the reference implementation, features are standardized per dimension using
    statistics of the training data before the principal subspace is estimated, which can be
    disabled through ``standardize``. The projection is applied to the centered representation,
    while the denominator uses the uncentered one.

    Scoring consumes one pooled feature vector per sample; grid-like feature maps of shape
    :math:`(B, C, H, W)` are rejected, because both the standardization statistics and the
    principal subspace are estimated over the feature axis of :math:`(N, D)` training vectors.

    Example Code:

    .. code :: python

        # WideResNet-40-2 has 128-dimensional penultimate features, so d must stay well below
        # that, and max-logit calibration is disabled for CNNs, as in the reference code
        model = WideResNet(num_classes=10)
        detector = NECO(encoder=model.features, head=None, d=32, use_max_logit=False)
        detector.fit(train_loader)
        scores = detector(images)

    :see Paper:
        `ArXiv <https://arxiv.org/abs/2310.06823>`__
    :see Implementation:
        `GitLab <https://gitlab.com/drti/neco>`__
    """

    requires_fit = True

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`: the dimension ``d`` of the
    #: estimated ETF subspace. The range brackets the values reported as optimal in the paper.
    #: Candidates larger than a model's feature dimension are clamped during fitting, so this
    #: single space is safe across architectures of different width.
    hyperparameter_space = {"d": [16, 32, 64, 100, 128, 256, 512, 768]}

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        head: Optional[Callable[[Tensor], Tensor]],
        d: int = 100,
        standardize: bool = True,
        use_max_logit: bool = True,
    ):
        """
        :param encoder: maps inputs to penultimate-layer features of shape :math:`(B, D)`. Can be
            ``None`` when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param head: maps features to logits. Required for ``use_max_logit=True``, otherwise it
            can be ``None``.
        :param d: dimension of the estimated ETF subspace. It has to stay well below the feature
            dimension :math:`D`, otherwise the projection retains nearly all of the representation
            and the ratio stops discriminating. The default follows the reference implementation,
            which used it for 768-dimensional features. As a rule of thumb, the paper suggests the
            number of principal components explaining 90% of the in-distribution training variance,
            which can be read off :attr:`explained_variance` after fitting. Values exceeding
            :math:`D` are clamped.
        :param standardize: standardize features to zero mean and unit variance (per dimension)
            using statistics of the training data, before estimating and applying the principal
            subspace. Disabling this can help when the in-distribution class clusters are already
            well separated.
        :param use_max_logit: multiply the subspace ratio with the maximum logit. The paper applies
            this calibration for transformer backbones (ViT, Swin, DeiT) and omits it for ResNets.
            Note that models which produce negative maximum logits invert the ordering of the
            affected samples, as in the original formulation.
        """
        super(NECO, self).__init__()

        self.encoder = encoder
        self.head = head
        self.d = d
        self.standardize = standardize
        self.use_max_logit = use_max_logit

        if use_max_logit and head is None:
            raise ModelNotSetException(msg="When using use_max_logit=True, head must not be None")

        #: per-dimension mean used for standardization, or ``None``
        self.feature_mean: Optional[Tensor] = None
        #: per-dimension standard deviation used for standardization, or ``None``
        self.feature_std: Optional[Tensor] = None
        #: mean of the (standardized) training features, subtracted before projecting
        self.pca_mean: Optional[Tensor] = None
        #: eigenvectors of the training covariance, sorted by decreasing eigenvalue, shape (D, D)
        self.components: Optional[Tensor] = None
        #: variance explained by each entry of :attr:`components`, in decreasing order, shape (D,)
        self.explained_variance: Optional[Tensor] = None

    @property
    def d(self) -> int:
        """
        Dimension of the estimated ETF subspace. Since fitting retains the full eigenbasis, this
        can be re-assigned on a fitted detector to try another subspace size without refitting.
        Values larger than the feature dimension are clamped, so that this attribute always
        describes the projection that is actually computed.
        """
        return self._d

    @d.setter
    def d(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"d must be >= 1, got {value}")

        components = getattr(self, "components", None)
        if components is not None and value > components.shape[1]:
            log.warning(
                f"d={value} exceeds the feature dimension {components.shape[1]} "
                f"and will be clamped."
            )
            value = components.shape[1]

        self._d = value

    def __repr__(self):
        return (
            f"NECO(d={self.d}, standardize={self.standardize}, use_max_logit={self.use_max_logit})"
        )

    def _standardize(self, z: Tensor) -> Tensor:
        """
        Apply the training-set standardization, if enabled.
        """
        if not self.standardize:
            return z

        return (z - self.feature_mean) / self.feature_std

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extracts features and estimates the principal subspace. Ignores OOD samples.

        :param data_loader: dataset to fit on, usually the training data
        """
        if self.encoder is None:
            raise ModelNotSetException

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader=data_loader, model=self.encoder, device=device)
        return self.fit_features(z, y)

    @torch.no_grad()
    def fit_features(self: Self, z: Tensor, y: Tensor, batch_size: int = 8192) -> Self:
        """
        Estimates standardization statistics and the principal subspace of the training features.
        Ignores OOD samples.

        :param z: training features of shape :math:`(N, D)`
        :param y: corresponding class labels. Samples with a label < 0 are ignored.
        :param batch_size: chunk size used while accumulating the covariance matrix, so that peak
            memory on the detector's device scales with ``batch_size`` instead of :math:`N`
        """
        if z.ndim != 2:
            raise ValueError(f"Expected features of shape (N, D), got {tuple(z.shape)}")

        known = is_known(y)
        if not known.any():
            raise ValueError("No ID samples")

        device = self.device or z.device
        z = z[known].detach().float()

        n, n_features = z.shape
        if n < 2:
            raise ValueError("At least two ID samples are required to estimate the subspace")

        # All fit-time reductions accumulate in float64, and chunks are cast before being
        # centered so that the subtraction cannot cancel catastrophically. At ImageNet scale
        # (n > 1e6) a float32 accumulator is already off by ~1e-6 absolute on the mean, and the
        # standardization is applied first, so its error would propagate into the covariance.
        # The resulting statistics are stored as float32, which is ample for applying them to a
        # single sample. Note that this cannot recover precision the *inputs* never had: float32
        # features with a very large mean offset are resolution-limited before fitting starts.
        if self.standardize:
            total = torch.zeros(n_features, dtype=torch.float64, device=device)
            for start in range(0, n, batch_size):
                total += z[start : start + batch_size].to(device).double().sum(dim=0)
            mean64 = total / n

            squares = torch.zeros(n_features, dtype=torch.float64, device=device)
            for start in range(0, n, batch_size):
                block = z[start : start + batch_size].to(device).double() - mean64
                squares += block.square().sum(dim=0)
            # ddof=0, as in sklearn's StandardScaler. Dimensions without variance would divide
            # by (almost) zero, so their scale is replaced by one -- the same substitution that
            # sklearn's _handle_zeros_in_scale performs.
            std64 = (squares / n).sqrt()
            std64 = torch.where(std64 > 1e-12, std64, torch.ones_like(std64))

            self.feature_mean = mean64.float()
            self.feature_std = std64.float()
        else:
            mean64 = None
            std64 = None
            self.feature_mean = None
            self.feature_std = None

        def standardize64(block: Tensor) -> Tensor:
            """Standardize a chunk in float64, so that centering cannot cancel catastrophically."""
            block = block.to(device).double()
            if not self.standardize:
                return block
            return (block - mean64) / std64

        # Mean of the (standardized) training features. The reference implementation projects with
        # sklearn's PCA, which subtracts it, so we do the same; the formula as printed in the paper
        # does not show this centering. Under standardize=True the two agree anyway, because
        # standardized training features have zero mean. Under standardize=False the centering does
        # shift the score for features with a large mean offset, such as pooled ReLU activations.
        total = torch.zeros(n_features, dtype=torch.float64, device=device)
        for start in range(0, n, batch_size):
            total += standardize64(z[start : start + batch_size]).sum(dim=0)
        pca_mean64 = total / n
        self.pca_mean = pca_mean64.float()

        # The eigenvectors of the covariance are the principal components. The covariance is only
        # (D, D), so accumulating it chunk-wise in double precision is cheap and keeps the peak
        # memory independent of the number of training samples.
        cov = torch.zeros(n_features, n_features, dtype=torch.float64, device=device)
        for start in range(0, n, batch_size):
            block = standardize64(z[start : start + batch_size]) - pca_mean64
            cov += block.T @ block
        cov /= n - 1

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # eigh returns ascending eigenvalues; NECO uses the *largest* ones first
        order = torch.argsort(eigenvalues, descending=True)
        self.components = eigenvectors[:, order].contiguous().float()
        self.explained_variance = eigenvalues[order].contiguous().float()

        # re-assigning through the property clamps d to the feature dimension, so that the
        # detector never reports a d that did not describe the computed projection
        self.d = self._d

        # n samples span at most n - 1 dimensions after centering; beyond that the eigenvectors
        # describe numerical noise rather than in-distribution structure. sklearn's PCA refuses
        # such a request outright; we warn and continue, since a small fit set is still usable.
        usable = min(n - 1, n_features)
        if self.d > usable:
            log.warning(
                f"d={self.d} exceeds the rank of the fitting data ({usable} for {n} samples "
                f"of dimension {n_features}); the extra directions carry no signal."
            )

        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model input, will be passed through the encoder
        """
        if self.encoder is None:
            raise ModelNotSetException

        if self.components is None:
            raise RequiresFittingException()

        device = self.device
        if device is not None:
            x = x.to(device)

        with torch.no_grad():
            z = self.encoder(x)

        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        :param z: penultimate-layer features of shape :math:`(N, D)`
        """
        if self.components is None:
            raise RequiresFittingException()

        if z.ndim != 2:
            raise ValueError(f"Expected features of shape (N, D), got {tuple(z.shape)}")

        device = self.components.device
        z = z.detach().to(device).float()

        scaled = self._standardize(z)

        # self.d is kept <= the feature dimension by its property setter and by fit_features
        projected = (scaled - self.pca_mean) @ self.components[:, : self.d]

        # Relative norm of the sample inside the estimated ETF subspace. Numerator and denominator
        # use the same (standardized) representation, which corresponds to the ViT and ResNet paths
        # of the reference implementation. Its DeiT and Swin paths instead take the denominator
        # from the raw features, which mixes two scalings and is not reproduced here; the results
        # reported for those two backbones were obtained that way.
        score = projected.norm(dim=-1) / scaled.norm(dim=-1).clamp(min=1e-12)

        if self.use_max_logit:
            if self.head is None:
                raise ModelNotSetException(
                    msg="When using use_max_logit=True, head must not be None"
                )

            if isinstance(self.head, torch.nn.Module):
                self.head.to(device)

            # logits are computed from the original, un-standardized features
            score = score * self.head(z).max(dim=-1).values

        # large values indicate inliers, so negate to follow the library convention
        return -score
