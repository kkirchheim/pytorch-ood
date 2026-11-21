"""

..  autoclass:: pytorch_ood.detector.Forte
    :members:
    :inherited-members:
    :show-inheritance:

"""

from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils.neighbors import _check_k
from ..utils.utils import _id_batches, _linalg_dtype

Encoder = Callable[[Tensor], Tensor]

_METHODS = ("gmm", "kde", "ocsvm")
# candidates of the reference implementation
_GMM_COMPONENTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_OCSVM_NU = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
# rows per block of pairwise distances
_CHUNK = 4096


def _sklearn(what: str):
    try:
        import sklearn  # optional dependency
    except ImportError as e:  # pragma: no cover - exercised only without scikit-learn
        raise ImportError(
            f"{what} requires the 'scikit-learn' package (pip install scikit-learn)."
        ) from e
    return sklearn


def _knn_radii(z: Tensor, k: int) -> Tensor:
    """
    Distance of each row of ``z`` to its ``k``-th nearest neighbor among the other rows. With
    ``k`` rows or fewer, the distance to the farthest row.
    """
    # the smallest distance is the one of a row to itself
    kth = min(k, z.shape[0] - 1) + 1
    return torch.cat(
        [torch.cdist(block, z).kthvalue(kth, dim=1).values for block in z.split(_CHUNK)]
    )


def _prdc(reference: Tensor, reference_radii: Tensor, z: Tensor, k: int) -> Tensor:
    """
    Per-point recall, density, precision and coverage of the sample ``z`` with respect to the
    reference sample, in this order.

    :param reference: reference sample of shape :math:`N \\times D`
    :param reference_radii: :func:`_knn_radii` of the reference sample
    :param z: evaluated sample of shape :math:`M \\times D`
    :param k: number of neighbors
    :return: statistics of shape :math:`M \\times 4`
    """
    n = reference.shape[0]
    radii = _knn_radii(z, k)
    statistics = []
    for block, block_radii in zip(z.split(_CHUNK), radii.split(_CHUNK)):
        distances = torch.cdist(reference, block)
        in_reference_balls = distances < reference_radii[:, None]
        in_own_ball = distances < block_radii
        statistics.append(
            torch.stack(
                [
                    in_own_ball.sum(dim=0).to(z.dtype) / n,
                    in_reference_balls.sum(dim=0).to(z.dtype) / (k * n),
                    in_reference_balls.any(dim=0).to(z.dtype),
                    (distances.min(dim=0).values < block_radii).to(z.dtype),
                ],
                dim=1,
            )
        )
    return torch.cat(statistics)


def _fit_density(statistics: np.ndarray, method: str, seed: int):
    """
    Fits the density model on statistics of shape :math:`N \\times S`, with the model selection of
    the reference implementation, which uses the fitted data only.
    """
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        best, best_score = None, -float("inf")
        for n_components in _GMM_COMPONENTS:
            if n_components >= len(statistics):
                continue
            gmm = GaussianMixture(n_components=n_components, random_state=seed).fit(statistics)
            score = gmm.score(statistics)
            if score > best_score:
                best, best_score = gmm, score
        return best

    if method == "kde":
        from scipy.stats import gaussian_kde

        kdes = [gaussian_kde(statistics.T, bw_method=bw) for bw in ("scott", "silverman")]
        scores = [kde.logpdf(statistics.T).mean() for kde in kdes]
        return kdes[1] if scores[1] > scores[0] else kdes[0]

    from sklearn.svm import OneClassSVM

    best, best_ratio = None, -float("inf")
    for nu in _OCSVM_NU:
        svm = OneClassSVM(kernel="rbf", gamma="auto", nu=nu).fit(statistics)
        ratio = (svm.predict(statistics) == 1).mean()
        if ratio > best_ratio:
            best, best_ratio = svm, ratio
    return best


def _typicality(density, statistics: np.ndarray, method: str) -> np.ndarray:
    if method == "gmm":
        return density.score_samples(statistics)
    if method == "kde":
        return density.logpdf(statistics.T)
    return density.decision_function(statistics)


class Forte(FeaturesDetector):
    """
    Implements the detector from the paper
    *Forte: Finding Outliers with Representation Typicality Estimation*.

    Forte compares a set of inputs, as a sample, to a reference sample of ID data. It works on
    the features of one or several encoders, usually self-supervised ones that were not trained
    on the ID data; the paper uses CLIP, ViT-MSN and DINOv2.

    For a reference sample :math:`R` and an evaluated sample :math:`T` of features, let
    :math:`r_k(z, S)` be the Euclidean distance of :math:`z` to its :math:`k`-th nearest
    neighbor among the other points of :math:`S`. Each :math:`z \\in T` is described by four
    statistics:

    .. math::
        \\begin{aligned}
        \\text{recall}(z) &= \\frac{1}{\\lvert R \\rvert} \\sum_{u \\in R} \\mathbb{1} \\lbrack \\lVert z - u \\rVert_2 < r_k(z, T) \\rbrack \\\\
        \\text{density}(z) &= \\frac{1}{k \\lvert R \\rvert} \\sum_{u \\in R} \\mathbb{1} \\lbrack \\lVert z - u \\rVert_2 < r_k(u, R) \\rbrack \\\\
        \\text{precision}(z) &= \\mathbb{1} \\lbrack \\exists u \\in R: \\lVert z - u \\rVert_2 < r_k(u, R) \\rbrack \\\\
        \\text{coverage}(z) &= \\mathbb{1} \\lbrack \\min_{u \\in R} \\lVert z - u \\rVert_2 < r_k(z, T) \\rbrack
        \\end{aligned}

    With several encoders, the statistics are computed for the features of each encoder and
    concatenated.

    Fitting splits the ID features into two halves. The first half is the reference sample
    :math:`R`. The statistics of the second half with respect to :math:`R` are used to fit a
    density model :math:`p`, see ``method``. The outlier score of :math:`z` is
    :math:`- \\log p` of its statistics (the negative decision function for the one-class SVM).
    The paper uses the log-density as a typicality score, which is larger for ID samples; the
    sign is flipped here.

    .. warning::
        Forte scores samples, not single inputs: all inputs of a call to :meth:`predict` or
        :meth:`predict_features` form the evaluated sample :math:`T`, and recall and coverage of
        an input depend on the other inputs of that call (see
        :attr:`~pytorch_ood.api.Detector.batch_dependent`). Pass all
        inputs you want to test in one call instead of scoring them in batches. An evaluated
        sample with :math:`k` points or fewer uses the distance to the farthest of its points as
        :math:`r_k`, so that a single input always has zero recall and coverage.

    The distances are computed in double precision (single precision on MPS, which does not
    support double precision).

    Fitting needs the optional dependency `scikit-learn <https://scikit-learn.org>`__.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.detector import Forte

        encoders = [torch.nn.Linear(10, 16), torch.nn.Linear(10, 32)]
        detector = Forte(encoders)
        # one feature tensor per encoder
        train_features = [torch.randn(200, 16), torch.randn(200, 32)]
        detector.fit_features(train_features, torch.zeros(200, dtype=torch.long))
        # the 64 inputs are scored as one sample
        scores = detector(torch.randn(64, 10))
    """

    info = DetectorInfo(
        paper=Paper(
            title="Forte: Finding Outliers with Representation Typicality Estimation",
            venue="ICLR",
            year=2025,
            url="https://openreview.net/forum?id=7XNgVPxCiA",
            code="https://github.com/DebarghaG/forte",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    requires_fit = True
    batch_dependent = True

    def __init__(
        self,
        encoder: Union[Encoder, Sequence[Encoder], None],
        method: str = "gmm",
        k: int = 5,
        seed: int = 0,
    ):
        """
        :param encoder: feature encoder, or a list of encoders that are applied to the same
            inputs. Can be ``None`` when using ``fit_features(...)`` and ``predict_features(...)``
            directly.
        :param method: density model fitted on the statistics of ID data. Its hyperparameter is
            selected on the fitted data, as in the reference implementation:

            - ``"gmm"``: Gaussian mixture model. The number of components is the one of 1, 2, 4,
              ..., 256 (and below the number of fitted statistics) with the highest likelihood.
            - ``"kde"``: Gaussian kernel density estimate. The bandwidth is the one of Scott's
              and Silverman's rule with the highest likelihood.
            - ``"ocsvm"``: one-class SVM with RBF kernel. :math:`\\nu` is the candidate with the
              largest fraction of inliers.
        :param k: number of neighbors :math:`k`
        :param seed: seed for the split of the fitting data and for the initialization of the
            Gaussian mixture model
        :raises ValueError: if ``method`` is unknown or ``k`` is below 1
        :raises TypeError: if ``k`` is not an integer
        """
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        _check_k(k)
        self.encoder = encoder
        self.method = method
        self.k = k
        self.seed = seed
        # fitted state, one entry per encoder
        self._reference: Optional[List[Tensor]] = None
        self._radii: Optional[List[Tensor]] = None
        self._density = None

    def _encoders(self) -> List[Encoder]:
        encoder = self._require_producer()
        return list(encoder) if isinstance(encoder, (list, tuple)) else [encoder]

    @staticmethod
    def _spaces(z: Union[Tensor, Sequence[Tensor]]) -> List[Tensor]:
        """The features per encoder, flattened to :math:`N \\times D`."""
        spaces = list(z) if isinstance(z, (list, tuple)) else [z]
        if len({len(t) for t in spaces}) != 1:
            raise ValueError(
                f"Expected features of the same inputs, got {[len(t) for t in spaces]} rows"
            )
        return [t.detach().flatten(1) for t in spaces]

    def _statistics(self, spaces: List[Tensor]) -> np.ndarray:
        """Statistics of shape :math:`N \\times 4E` for :math:`E` encoders."""
        if len(spaces) != len(self._reference):
            raise ValueError(
                f"Fitted on the features of {len(self._reference)} encoder(s), "
                f"got the features of {len(spaces)}"
            )
        statistics = [
            _prdc(reference, radii, z.to(reference.device, reference.dtype), self.k)
            for reference, radii, z in zip(self._reference, self._radii, spaces)
        ]
        return torch.cat(statistics, dim=1).double().cpu().numpy()

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extract features of the ID samples in a loader with each encoder and forward them to
        ``fit_features(...)``.

        :param data_loader: loader yielding batches ``(x, y)``; OOD samples are ignored
        :return: the fitted detector
        :raises ModelNotSetException: if the detector has no encoder
        :raises ValueError: if the loader contains no ID samples
        """
        encoders = self._encoders()
        device = self._fit_device()

        # one pass, so that the rows of all encoders belong to the same inputs
        spaces, labels = [[] for _ in encoders], []
        with torch.no_grad():
            for x, y in _id_batches(data_loader):
                x = x.to(device)
                for space, encoder in zip(spaces, encoders):
                    space.append(encoder(x).flatten(1))
                labels.append(y)
        return self.fit_features([torch.cat(space) for space in spaces], torch.cat(labels))

    def fit_features(self, z: Union[Tensor, Sequence[Tensor]], labels: Tensor) -> Self:
        """
        Splits the features into the reference sample and the sample the density model is fitted
        on. Ignores OOD inputs.

        :param z: features of shape :math:`N \\times D`, or one such tensor per encoder, with rows
            for the same inputs
        :param labels: labels for features, shape :math:`N`; only used to find the ID samples
        :return: the fitted detector
        :raises ValueError: if there are fewer than 4 ID samples
        """
        train_test_split = _sklearn("Forte").model_selection.train_test_split

        spaces = self._spaces(z)
        n = len(spaces[0])
        if n < 4:
            raise ValueError(f"Forte needs at least 4 ID samples to fit on, got {n}")
        device = self.device or spaces[0].device
        dtype = _linalg_dtype(device)
        # the same rows for every encoder
        reference_rows, fit_rows = train_test_split(
            np.arange(n), test_size=0.5, random_state=self.seed
        )

        self._reference = [z[reference_rows].to(device, dtype) for z in spaces]
        self._radii = [_knn_radii(reference, self.k) for reference in self._reference]
        statistics = self._statistics([z[fit_rows] for z in spaces])
        self._density = _fit_density(statistics, self.method, self.seed)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Apply each encoder and forward the features to ``predict_features(...)``.

        :param x: inputs that form the evaluated sample, e.g. images of shape
            :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if the detector has no encoder
        :raises RequiresFittingException: if the detector was not fitted
        """
        encoders = self._encoders()
        device = self.device
        if device is not None:
            x = x.to(device)
        with torch.no_grad():
            spaces = [encoder(x) for encoder in encoders]
        return self.predict_features(spaces)

    def predict_features(self, z: Union[Tensor, Sequence[Tensor]]) -> Tensor:
        """
        :param z: features of the evaluated sample of shape :math:`B \\times D`, or one such
            tensor per encoder
        :return: outlier scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        :raises ValueError: if the number of feature tensors differs from the fitted one
        """
        if self._density is None:
            raise RequiresFittingException()

        spaces = self._spaces(z)
        typicality = _typicality(self._density, self._statistics(spaces), self.method)
        return -torch.as_tensor(
            typicality,
            device=self._reference[0].device,
            dtype=torch.promote_types(spaces[0].dtype, torch.float32),
        )
