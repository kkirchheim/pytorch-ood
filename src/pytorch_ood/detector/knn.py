"""

..  autoclass:: pytorch_ood.detector.KNN
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional

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
from ..utils import extract_features, is_known
from ..utils.neighbors import NearestNeighborSearch, TorchNearestNeighbors

log = logging.getLogger(__name__)


class KNN(FeaturesDetector):
    """
    Implements the detector from the paper
    *Out-of-Distribution Detection with Deep Nearest Neighbors*.

    The features of the fitted ID samples and of each input are normalized to unit length (see
    ``normalize``). The outlier score of an input is the Euclidean distance of its normalized
    feature :math:`z` to the :math:`k`-th nearest normalized feature of the fitted data:

    .. math:: \\lVert z - z_{(k)} \\rVert_2 \\quad \\text{with} \\quad z = \\frac{f(x)}{\\lVert f(x) \\rVert_2}

    where :math:`f` is the ``encoder`` and :math:`z_{(1)}, \\dots, z_{(n)}` are the normalized features of
    the fitted data, sorted by increasing distance to :math:`z`.

    The neighbors are found by ``search``, by default an exact search with
    :class:`TorchNearestNeighbors <pytorch_ood.utils.TorchNearestNeighbors>`. It keeps the
    features of the fitted data in memory (:math:`n \\times D` values), on the detector's device:
    :meth:`to` moves them. For a fitted set too large for the GPU, pin it to the CPU with
    ``search=TorchNearestNeighbors(device="cpu")``.

    The original paper found that using contrastive pre-training could increase the performance.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Out-of-Distribution Detection with Deep Nearest Neighbors",
            venue="ICML",
            year=2022,
            url="https://proceedings.mlr.press/v162/sun22d.html",
            code="https://github.com/deeplearning-wisc/knn-ood",
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    # matches the ``K`` sweep used by OpenOOD
    #: Default search space for :class:`pytorch_ood.utils.GridSearch`.
    hyperparameter_space = {"k": [50, 100, 200, 500, 1000]}

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        k: int = 50,
        normalize: bool = True,
        search: Optional[NearestNeighborSearch] = None,
    ):
        """
        :param encoder: feature encoder :math:`f`. Can be ``None`` when using
            ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param k: number of neighbors :math:`k`; the score is the distance to the :math:`k`-th
            nearest neighbor. The best value depends on the dataset, see
            ``hyperparameter_space``. It can be changed after fitting.
        :param normalize: whether to normalize the features to unit length,
            :math:`z = f(x) / \\lVert f(x) \\rVert_2`. If ``False``, :math:`z = f(x)`, and the
            distances also depend on the feature norms.
        :param search: nearest-neighbor search with ``metric == "sqeuclidean"``. Defaults to
            :class:`TorchNearestNeighbors() <pytorch_ood.utils.TorchNearestNeighbors>`.
        :raises ValueError: if ``search`` uses another metric
        """
        # k = 50 is the paper's choice for CIFAR-10 (200 for CIFAR-100, 1000 for ImageNet)
        if search is None:
            search = TorchNearestNeighbors()
        if search.metric != "sqeuclidean":
            raise ValueError(
                f"KNN needs a search with metric 'sqeuclidean', got {search.metric!r}"
            )
        self.encoder = encoder
        self.k = k
        self.normalize = normalize
        self.search = search
        self._n_fitted: Optional[int] = None

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: inputs, will be passed through ``encoder``
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if the detector has no ``encoder``
        :raises RequiresFittingException: if the detector was not fitted
        """
        if self.encoder is None:
            raise ModelNotSetException()

        device = self.device
        if device is not None:
            x = x.to(device)

        z = self.encoder(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        :param z: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`, in the dtype of ``z``
        :raises RequiresFittingException: if the detector was not fitted
        :raises ValueError: if ``k`` exceeds the number of fitted samples
        """
        if self._n_fitted is None:
            raise RequiresFittingException()
        self._check_k(self._n_fitted, "fitted samples")

        # squared distances, nearest first; the k-th is the last column
        values, _ = self.search.search(self._prepare(z), self.k)
        return values[:, -1].sqrt()

    def fit_features(self, z: Tensor, labels: Tensor) -> Self:
        """
        Fits the nearest-neighbor search. Ignores OOD inputs.

        :param z: features of shape :math:`N \\times D`
        :param labels: labels for features, shape :math:`N`
        :return: the fitted detector
        :raises ValueError: if ``labels`` contains fewer than ``k`` in-distribution samples
        """
        known = is_known(labels).to(z.device)

        if not known.any():
            raise ValueError("No ID samples")
        n_known = int(known.sum())
        self._check_k(n_known, "ID samples")

        self.search.fit(self._prepare(z.detach()[known]))
        # an unpinned backend follows the detector, like its other state
        device = self.device
        if device is not None:
            self.search = self._move_value_to_device(self.search, device)
        self._n_fitted = n_known

        return self

    def _check_k(self, n: int, what: str) -> None:
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds the number of {what} ({n})")

    def _prepare(self, z: Tensor) -> Tensor:
        if not self.normalize:
            return z
        # the paper's normalization; the clamp keeps zero vectors finite
        return z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extracts features and fits the nearest-neighbor search.

        :param data_loader: data loader. OOD inputs will be ignored.
        :return: the fitted detector
        :raises ModelNotSetException: if the detector has no ``encoder``
        """
        if self.encoder is None:
            raise ModelNotSetException()

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(model=self.encoder, data_loader=data_loader, device=device)
        return self.fit_features(z, y)
