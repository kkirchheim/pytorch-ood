"""

..  autoclass:: pytorch_ood.utils.NearestNeighborSearch
    :members:

..  autoclass:: pytorch_ood.utils.TorchNearestNeighbors
    :members:

"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal, Self

from ..api import RequiresFittingException

__all__ = ["NearestNeighborSearch", "TorchNearestNeighbors"]

# block budget of the "auto" batch size off the GPU
_CPU_BLOCK_BYTES = 1 << 30

_METRICS = ("sqeuclidean", "inner_product")


class NearestNeighborSearch(ABC):
    """
    Base class of nearest-neighbor searches: :meth:`fit` stores a reference set, :meth:`search`
    finds the :math:`k` nearest reference samples of queries. Subclass it to plug in another
    search, for example a wrapper around an approximate search library.

    .. rubric:: Examples

    .. code-block:: python

        import torch
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


        search = CdistSearch().fit(torch.randn(100, 16))
        distances, indices = search.search(torch.randn(8, 16), k=5)  # both 8 x 5
    """

    #: What the values returned by :meth:`search` measure. ``"sqeuclidean"``: squared Euclidean
    #: distances, ascending (nearest first). ``"inner_product"``: inner products, descending
    #: (largest first).
    metric: str

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
        Find the :math:`k` nearest reference samples of each query, for :math:`1 \\le k \\le N`,
        where the nearest have the smallest distances or the largest inner products (see
        ``metric``).

        :param queries: queries of shape :math:`B \\times D`
        :param k: number of neighbors :math:`k`
        :return: the values and the indices of the neighbors, both of shape
            :math:`B \\times k` and on the device of ``queries``. The values are in units of
            ``metric``, ordered from the nearest neighbor to the :math:`k`-th, and in the dtype
            of ``queries``. The indices refer to the reference set.
        """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """
        Number of samples in the reference set, :math:`N`; 0 before fitting.
        """

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move the search to ``device`` in place. The default does nothing; override it if the
        search holds tensors.

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

    and the inner product as :math:`q^\\top r`, in blocks of ``batch_size`` queries, so that a
    block needs memory for a ``batch_size`` :math:`\\times N` matrix of values. The reference set
    itself stays in memory (:math:`N \\times D` values on its device).

    Values are computed in the dtype of the reference set. In ``float32``, the distances of very
    close neighbors lose relative precision, more so for samples with large norms; use
    ``float64`` samples if that matters.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.utils import TorchNearestNeighbors

        search = TorchNearestNeighbors(batch_size=256).fit(torch.randn(1000, 16))
        # squared distances to the 5 nearest reference samples, nearest first, and their indices
        distances, indices = search.search(torch.randn(8, 16), k=5)

        # the 5 largest inner products instead, largest first
        search = TorchNearestNeighbors(metric="inner_product").fit(torch.randn(1000, 16))
        similarities, indices = search.search(torch.randn(8, 16), k=5)
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: Union[int, Literal["auto"], None] = "auto",
        metric: Literal["sqeuclidean", "inner_product"] = "sqeuclidean",
    ):
        """
        :param device: device to keep the reference set on. If given, the reference set is
            pinned there: :meth:`to` does not move it, queries on other devices are moved to it,
            and the results are moved back. If ``None``, the reference set stays where it was
            fitted and :meth:`to` moves it.
        :param batch_size: number of queries per block. ``"auto"`` derives it on each call from
            the free GPU memory, or from a fixed budget of 1 GiB on other devices. ``None``
            processes all queries in one block, which needs memory for :math:`B \\times N`
            values.
        :param metric: ``"sqeuclidean"`` for squared Euclidean distances, ascending, or
            ``"inner_product"`` for inner products, descending
        :raises ValueError: if ``batch_size`` is not ``"auto"``, ``None`` or a positive integer,
            or ``metric`` is not one of the above
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
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        #: ``"sqeuclidean"`` or ``"inner_product"``, see :attr:`NearestNeighborSearch.metric`.
        self.metric = metric
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
        :return: the fitted search
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
        :return: squared Euclidean distances of shape :math:`B \\times k`, ascending, or inner
            products, descending (see ``metric``), and the indices of the neighbors in the
            reference set, of shape :math:`B \\times k`. Both on the device of ``queries``, the
            values in their dtype.
        :raises RequiresFittingException: if the search was not fitted
        :raises ValueError: if ``queries`` does not have shape :math:`B \\times D` with the
            :math:`D` of the reference set, or ``k`` is not between 1 and the size of the
            reference set
        """
        if self._reference is None:
            raise RequiresFittingException()
        reference = self._reference
        n, d = reference.shape
        if queries.dim() != 2 or queries.shape[1] != d:
            raise ValueError(f"queries must have shape B x {d}, got {tuple(queries.shape)}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got k={k}")
        if k > n:
            raise ValueError(f"k={k} exceeds the number of reference samples ({n})")

        # queries are cast to the reference dtype (cheap), not the reference to the queries'
        # dtype, which would copy the whole reference set
        q = queries.detach().to(device=reference.device, dtype=reference.dtype)
        euclidean = self.metric == "sqeuclidean"
        if euclidean:
            # per call rather than cached at fit: costs 1/B of the distance computation
            squared_norms = torch.einsum("nd,nd->n", reference, reference)
        rows = self._rows_per_block(q.shape[0], n, reference)
        values_list, indices_list = [], []
        # an empty batch is a single empty block
        for block in q.split(rows):
            if euclidean:
                # ||q||^2 + ||r||^2 - 2 q.r; cancellation can make it slightly negative for
                # (near-)duplicates, hence the clamp. The absolute error grows with the squared
                # norms. With TF32 matmuls enabled (torch.set_float32_matmul_precision("high")),
                # errors grow to about 1e-3.
                d2 = torch.addmm(squared_norms, block, reference.T, alpha=-2)
                d2.add_(torch.einsum("bd,bd->b", block, block)[:, None]).clamp_(min=0)
                v, i = d2.topk(k, dim=1, largest=False)
            else:
                v, i = (block @ reference.T).topk(k, dim=1)
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
            return max(1, n_queries)
        if self.batch_size != "auto":
            return self.batch_size
        if reference.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(reference.device)
            budget = free // 2
        else:
            budget = _CPU_BLOCK_BYTES
        # the distance block plus about as much workspace for topk
        return max(1, budget // (2 * n * reference.element_size()))
