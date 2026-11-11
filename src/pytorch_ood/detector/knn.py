"""

..  autoclass:: pytorch_ood.detector.KNN
    :members:
    :inherited-members:
    :show-inheritance:

"""

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils.neighbors import NearestNeighborSearch, TorchNearestNeighbors, _check_k, _check_search


class KNN(FeaturesDetector):
    """
    Implements the detector from the paper
    *Out-of-Distribution Detection with Deep Nearest Neighbors*.

    The features of the fitted ID samples and of each input are normalized to unit length (see
    ``normalize``). The outlier score of an input is the Euclidean distance of its normalized
    feature :math:`z` to the :math:`k`-th nearest normalized feature of the fitted data:

    .. math:: \\lVert z - z_{(k)} \\rVert_2 \\quad \\text{with} \\quad z = \\frac{f(x)}{\\lVert f(x) \\rVert_2}

    where :math:`f` is the ``encoder`` and :math:`z_{(1)}, \\dots, z_{(N)}` are the normalized features of
    the fitted data, sorted by increasing distance to :math:`z`. The paper uses the negative
    distance, which is larger for ID samples; the sign is flipped here.

    The neighbors are found by ``search``, by default an exact search with
    :class:`TorchNearestNeighbors <pytorch_ood.utils.TorchNearestNeighbors>`. That search keeps
    the features of the :math:`N` fitted samples in memory (:math:`N \\times D` values) on the
    detector's device, and :meth:`to` moves them. For a fitted set too large for the GPU, pin it
    to the CPU with ``search=TorchNearestNeighbors(device="cpu")``; the search then runs on the
    CPU.

    The original paper found that using contrastive pre-training could increase the performance.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.detector import KNN
        from pytorch_ood.utils import TorchNearestNeighbors

        encoder = torch.nn.Linear(10, 16)
        # the fitted features stay on the CPU, also if the detector is moved to a GPU
        detector = KNN(encoder, k=5, search=TorchNearestNeighbors(device="cpu"))
        detector.fit_features(torch.randn(100, 16), torch.zeros(100, dtype=torch.long))
        scores = detector(torch.randn(8, 10))
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
        :param search: instance of a
            :class:`NearestNeighborSearch <pytorch_ood.utils.NearestNeighborSearch>` with
            ``metric == "sqeuclidean"`` (see there for writing your own). Defaults to
            :class:`TorchNearestNeighbors() <pytorch_ood.utils.TorchNearestNeighbors>`. Fitting
            the detector fits this instance and replaces its reference set, so give each detector
            its own instance.
        :raises ValueError: if ``k`` is below 1 or ``search`` uses another metric
        :raises TypeError: if ``k`` is not an integer or ``search`` is not an instance of
            :class:`NearestNeighborSearch <pytorch_ood.utils.NearestNeighborSearch>`
        """
        # k = 50 is the paper's choice for CIFAR-10 (200 for CIFAR-100, 1000 for ImageNet)
        _check_k(k)
        self.k = k
        if search is None:
            search = TorchNearestNeighbors()
        _check_search(search, "sqeuclidean", "KNN")
        self.encoder = encoder
        self.normalize = normalize
        #: The nearest-neighbor search, fitted with the features of the ID samples.
        self.search = search
        self._is_fitted = False

    def predict_features(self, z: Tensor) -> Tensor:
        """
        :param z: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        :raises ValueError: if ``k`` exceeds the number of fitted samples
        """
        if not self._is_fitted:
            raise RequiresFittingException()
        _check_k(self.k, self.search.n_samples, "fitted samples")

        # squared distances, nearest first; the k-th is the last column
        values, _ = self.search.search(self._prepare(z), self.k)
        # the sqrt magnifies the search's float32 errors near 0: near-duplicates of fitted samples
        # (e.g. k=1 on training data) score around 1e-3 instead of 0. The reference code scores
        # with the squared distance (faiss IndexFlatL2); the sqrt follows the paper and does not
        # change the ranking.
        return values[:, -1].sqrt()

    def fit_features(self, z: Tensor, labels: Tensor) -> Self:
        """
        Fits the nearest-neighbor search. Ignores OOD inputs.

        :param z: features of shape :math:`N \\times D`
        :param labels: labels for features, shape :math:`N`
        :return: the fitted detector
        :raises ValueError: if ``labels`` contains no or fewer than ``k`` in-distribution
            samples
        """
        _check_k(self.k, len(labels), "ID samples")

        self.search.fit(self._prepare(z.detach()))
        # an unpinned backend follows the detector, like its other state
        device = self.device
        if device is not None:
            self.search.to(device)
        self._is_fitted = True

        return self

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move the detector's state to ``device``, including the fitted features of ``search``
        unless the search is pinned to a device.

        :param device: target device, e.g. ``"cuda:0"`` or a :class:`torch.device`
        :return: self
        """
        super().to(device)
        # in place; the return value is not needed
        self.search.to(device)
        return self

    def _prepare(self, z: Tensor) -> Tensor:
        if not self.normalize:
            return z
        # the paper's normalization; the clamp keeps zero vectors finite
        return z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
