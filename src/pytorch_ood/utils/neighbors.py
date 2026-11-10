"""

..  autoclass:: pytorch_ood.utils.NearestNeighborSearch
    :members:

..  autoclass:: pytorch_ood.utils.TorchNearestNeighbors
    :members:

"""

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal, Self

from ..api import RequiresFittingException

__all__ = ["NearestNeighborSearch", "TorchNearestNeighbors"]

# block budget of the "auto" batch size off the GPU
_CPU_BLOCK_BYTES = 1 << 30


class NearestNeighborSearch(ABC):
    """
    Base class of the nearest-neighbor search used by detectors such as
    :class:`KNN <pytorch_ood.detector.KNN>`. Subclass it to plug in another search, for example
    a wrapper around an approximate search library, and pass an instance as the detector's
    ``search``.

    The detector calls :meth:`fit` with its preprocessed features, which replaces the reference
    set of the instance. Give each detector its own instance: detectors sharing one instance
    score against whichever detector was fitted last.

    A minimal search for :class:`KNN <pytorch_ood.detector.KNN>`:

    .. code:: python

        import torch
        from pytorch_ood.detector import KNN
        from pytorch_ood.utils import NearestNeighborSearch

        class CdistSearch(NearestNeighborSearch):
            metric = "sqeuclidean"

            def fit(self, reference):
                self.reference = reference
                return self

            def search(self, queries, k):
                d2 = torch.cdist(queries, self.reference) ** 2
                return d2.topk(k, dim=1, largest=False)  # ascending, nearest first

            @property
            def n_samples(self):
                return len(self.reference)

        detector = KNN(encoder, search=CdistSearch())

    """

    #: What the values returned by :meth:`search` measure. ``"sqeuclidean"``: squared Euclidean
    #: distances, ascending (nearest first). This is the only metric
    #: :class:`KNN <pytorch_ood.detector.KNN>` accepts.
    metric: ClassVar[str]

    @abstractmethod
    def fit(self, reference: Tensor) -> Self:
        """
        Store the reference set.

        :param reference: reference samples of shape :math:`N \\times D`
        :return: the fitted search
        """

    @abstractmethod
    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Find the :math:`k` nearest reference samples of each query, for :math:`1 \\le k \\le N`.

        :param queries: queries of shape :math:`B \\times D`
        :param k: number of neighbors :math:`k`
        :return: the values and the indices of the neighbors, both of shape
            :math:`B \\times k` and on the device of ``queries``. The values are in units of
            ``metric`` (squared distances for ``"sqeuclidean"``), ordered from the nearest
            neighbor to the :math:`k`-th, and in the dtype of ``queries``. The indices refer to
            the reference set.
        """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """
        Number of samples in the reference set, :math:`N`.
        """

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move the search to ``device`` in place. The default does nothing; override it if the
        search holds tensors that should follow the detector.
        :meth:`KNN.to <pytorch_ood.detector.KNN.to>` calls it.

        :param device: target device
        :return: self
        """
        return self


class TorchNearestNeighbors(NearestNeighborSearch):
    """
    Exact nearest-neighbor search in PyTorch, by brute force on the device of the reference set.

    For a query :math:`q` and a reference sample :math:`r`, the squared Euclidean distance is
    computed as

    .. math:: \\lVert q \\rVert_2^2 + \\lVert r \\rVert_2^2 - 2 q^\\top r

    in blocks of ``batch_size`` queries, so that a block needs memory for a
    ``batch_size`` :math:`\\times N` matrix of distances. The reference set itself stays in memory
    (:math:`N \\times D` values on its device).

    Distances are computed in the dtype of the reference set. In ``float32``, the formula above
    has absolute errors proportional to :math:`\\lVert q \\rVert_2^2 + \\lVert r \\rVert_2^2`:
    around :math:`10^{-6}` for unit-length samples, more for larger ones. Very close neighbors
    lose relative precision; fit and query with ``float64`` samples if you need more.

    Example: keep a large reference set on the CPU while the detector runs on the GPU. The
    search then runs on the CPU:

    .. code:: python

        detector = KNN(model.features, search=TorchNearestNeighbors(device="cpu"))
        detector.to("cuda")  # moves the encoder, not the reference set

    """

    metric = "sqeuclidean"

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: Union[int, Literal["auto"], None] = "auto",
    ):
        """
        :param device: device to keep the reference set on. If given, the reference set is
            pinned there: :meth:`to` does not move it, queries on other devices are moved to it,
            and the results are moved back. If ``None``, the reference set stays where it was
            fitted and :meth:`to` moves it.
        :param batch_size: number of queries per block. ``"auto"`` derives it on each call from
            the free GPU memory, or from a fixed budget of 1 GiB on other devices. ``None``
            processes all queries in one block, which needs memory for :math:`B \\times N`
            distances.
        :raises ValueError: if ``batch_size`` is not ``"auto"``, ``None`` or a positive integer
        """
        valid_int = (
            isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size >= 1
        )
        if not (
            batch_size is None
            or valid_int
            or (isinstance(batch_size, str) and batch_size == "auto")
        ):
            raise ValueError(
                f"batch_size must be 'auto', None or a positive integer, got {batch_size!r}"
            )
        #: Device the reference set is pinned to, or ``None`` if it follows :meth:`to`.
        self.device = torch.device(device) if device is not None else None
        #: Number of queries per block: ``"auto"``, ``None`` (one block) or an integer.
        self.batch_size = batch_size
        self._reference: Optional[Tensor] = None

    @property
    def reference(self) -> Optional[Tensor]:
        """
        The fitted reference set of shape :math:`N \\times D`, or ``None`` before fitting.
        """
        return self._reference

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the reference set, :math:`N`; 0 before fitting.
        """
        return 0 if self._reference is None else self._reference.shape[0]

    def fit(self, reference: Tensor) -> Self:
        """
        Store the reference set, on ``device`` if one was given. The tensor is stored without
        copying unless it is moved to ``device``, so do not modify it in place afterwards.

        :param reference: reference samples of shape :math:`N \\times D`
        :return: the fitted backend
        :raises ValueError: if ``reference`` is not two-dimensional
        """
        if reference.dim() != 2:
            raise ValueError(f"reference must have shape N x D, got {tuple(reference.shape)}")
        reference = reference.detach()
        if self.device is not None:
            reference = reference.to(self.device)
        self._reference = reference
        return self

    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Find the :math:`k` nearest reference samples of each query.

        :param queries: queries of shape :math:`B \\times D`
        :param k: number of neighbors :math:`k`
        :return: squared Euclidean distances of shape :math:`B \\times k`, ascending, and the
            indices of the neighbors in the reference set, of shape :math:`B \\times k`. Both on
            the device of ``queries``, the distances in their dtype.
        :raises RequiresFittingException: if the backend was not fitted
        :raises ValueError: if ``k`` is not between 1 and the size of the reference set
        """
        if self._reference is None:
            raise RequiresFittingException()
        reference = self._reference
        n = reference.shape[0]
        if k < 1:
            raise ValueError(f"k must be at least 1, got k={k}")
        if k > n:
            raise ValueError(f"k={k} exceeds the number of reference samples ({n})")

        # queries are cast to the reference dtype (cheap), not the reference to the queries'
        # dtype, which would copy the whole reference set
        q = queries.detach().to(device=reference.device, dtype=reference.dtype)
        if q.shape[0] == 0:
            values = q.new_empty((0, k))
            indices = torch.empty((0, k), dtype=torch.long, device=q.device)
        else:
            # per call rather than cached at fit: costs 1/B of the distance computation
            squared_norms = torch.einsum("nd,nd->n", reference, reference)
            rows = self._rows_per_block(q.shape[0], n, reference)
            values_list, indices_list = [], []
            for block in q.split(rows):
                # ||q||^2 + ||r||^2 - 2 q.r; cancellation can make it slightly negative for
                # (near-)duplicates, hence the clamp. With TF32 matmuls enabled
                # (torch.set_float32_matmul_precision("high")), errors grow to about 1e-3.
                d2 = torch.addmm(squared_norms, block, reference.T, alpha=-2)
                d2.add_(torch.einsum("bd,bd->b", block, block)[:, None]).clamp_(min=0)
                v, i = d2.topk(k, dim=1, largest=False)
                values_list.append(v)
                indices_list.append(i)
            values, indices = torch.cat(values_list), torch.cat(indices_list)

        return values.to(device=queries.device, dtype=queries.dtype), indices.to(queries.device)

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move the reference set to ``device``, unless it is pinned (see ``device``).

        :param device: target device
        :return: self
        """
        if self.device is None and self._reference is not None:
            self._reference = self._reference.to(device)
        return self

    def _rows_per_block(self, n_queries: int, n: int, reference: Tensor) -> int:
        if self.batch_size is None:
            return n_queries
        if self.batch_size != "auto":
            return self.batch_size
        if reference.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(reference.device)
            budget = free // 2
        else:
            budget = _CPU_BLOCK_BYTES
        # the distance block plus about as much workspace for topk
        return max(1, budget // (2 * n * reference.element_size()))
