"""

..  autoclass:: pytorch_ood.utils.NearestNeighborSearch
    :members:

..  autoclass:: pytorch_ood.utils.TorchNearestNeighbors
    :members:

"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal, Protocol, Self, runtime_checkable

from ..api import RequiresFittingException

__all__ = ["NearestNeighborSearch", "TorchNearestNeighbors"]

# block budget of the "auto" batch size off the GPU
_CPU_BLOCK_BYTES = 1 << 30


@runtime_checkable
class NearestNeighborSearch(Protocol):
    """
    Interface of the nearest-neighbor search used by detectors such as
    :class:`KNN <pytorch_ood.detector.KNN>`. Any object with these members can be passed as a
    detector's ``search``, for example a wrapper around an approximate search library. The
    detector calls :meth:`fit` itself, with its already preprocessed features, so one instance
    belongs to one detector.

    A backend may also define ``to(device)``, which moves it in place; the detector's
    :meth:`to <pytorch_ood.detector.KNN.to>` calls it and ignores the return value. A backend
    without ``to`` stays where it is.

    A minimal backend for :class:`KNN <pytorch_ood.detector.KNN>`:

    .. code:: python

        class CdistSearch:
            metric = "sqeuclidean"

            def fit(self, reference):
                self.reference = reference
                return self

            def search(self, queries, k):
                d2 = torch.cdist(queries, self.reference) ** 2
                return d2.topk(k, dim=1, largest=False)  # ascending, nearest first

    """

    #: What the values returned by :meth:`search` measure. ``"sqeuclidean"``: squared Euclidean
    #: distances, ascending (nearest first). This is the only metric
    #: :class:`KNN <pytorch_ood.detector.KNN>` accepts.
    metric: str

    def fit(self, reference: Tensor) -> Self:
        """
        Store the reference set.

        :param reference: reference samples of shape :math:`N \\times D`
        :return: the fitted backend
        """
        ...

    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Find the :math:`k` nearest reference samples of each query.

        :param queries: queries of shape :math:`B \\times D`
        :param k: number of neighbors :math:`k`
        :return: values of shape :math:`B \\times k`, in units of ``metric`` (squared distances for
            ``"sqeuclidean"``) and ordered from the nearest neighbor to the :math:`k`-th, and the
            indices of the neighbors in the reference
            set, of shape :math:`B \\times k`. Both on the device of ``queries``, the values in
            their dtype.
        """
        ...


class TorchNearestNeighbors:
    """
    Exact nearest-neighbor search in PyTorch, by brute force on the device of the reference set.

    For a query :math:`q` and a reference sample :math:`r`, the squared Euclidean distance is
    computed as

    .. math:: \\lVert q \\rVert_2^2 + \\lVert r \\rVert_2^2 - 2 q^\\top r

    in blocks of ``batch_size`` queries, so that a block needs memory for a
    ``batch_size`` :math:`\\times N` matrix of distances. The reference set itself stays in memory
    (:math:`N \\times D` values on its device).

    Distances are computed in the dtype of the reference set. In ``float32``, the formula above
    has absolute errors around :math:`10^{-6}`, so very close neighbors lose relative precision;
    fit and query with ``float64`` samples if you need more.

    Example: keep a large reference set on the CPU while the detector runs on the GPU:

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
        self.device = torch.device(device) if device is not None else None
        self.batch_size = batch_size
        self._reference: Optional[Tensor] = None

    @property
    def reference(self) -> Optional[Tensor]:
        """
        The fitted reference set of shape :math:`N \\times D`, or ``None`` before fitting.
        """
        return self._reference

    def fit(self, reference: Tensor) -> Self:
        """
        Store the reference set, on ``device`` if one was given.

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
