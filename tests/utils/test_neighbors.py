import pytest
import torch

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.utils import NearestNeighborSearch, TorchNearestNeighbors
from tests.helpers import requires
from tests.helpers.components import code_blocks


def _reference_search(reference: torch.Tensor, queries: torch.Tensor, k: int):
    """Brute-force k nearest neighbours in float64: squared Euclidean distances, ascending."""
    d2 = torch.cdist(queries.double(), reference.double()) ** 2
    values, indices = d2.sort(dim=1)
    return values[:, :k], indices[:, :k]


@pytest.fixture
def data(seed):
    reference = torch.randn(300, 16, generator=seed, dtype=torch.float64)
    queries = torch.randn(40, 16, generator=seed, dtype=torch.float64)
    return reference, queries


def test_is_a_nearest_neighbor_search():
    search = TorchNearestNeighbors()
    assert isinstance(search, NearestNeighborSearch)
    assert search.metric == "sqeuclidean"
    # the class itself is not a search, only its instances
    assert not isinstance(TorchNearestNeighbors, NearestNeighborSearch)


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        NearestNeighborSearch()


def test_subclass_must_implement_n_samples():
    class NoSize(NearestNeighborSearch):
        metric = "sqeuclidean"

        def fit(self, reference):
            return self

        def search(self, queries, k):
            raise NotImplementedError

    with pytest.raises(TypeError, match="n_samples"):
        NoSize()


def test_n_samples(data):
    search = TorchNearestNeighbors()
    assert search.n_samples == 0
    search.fit(data[0])
    assert search.n_samples == 300
    search.fit(data[0][:7])
    assert search.n_samples == 7


def test_base_to_is_a_no_op(data):
    class Minimal(NearestNeighborSearch):
        metric = "sqeuclidean"

        def fit(self, reference):
            self.reference = reference
            return self

        def search(self, queries, k):
            return (torch.cdist(queries, self.reference) ** 2).topk(k, dim=1, largest=False)

        @property
        def n_samples(self):
            return len(self.reference)

    search = Minimal().fit(data[0])
    assert search.to("meta") is search
    assert search.reference.device.type == "cpu"


def test_fit_rejects_non_2d_reference():
    with pytest.raises(ValueError, match="N x D"):
        TorchNearestNeighbors().fit(torch.randn(5))


@pytest.mark.parametrize("k", [1, 5, 50, 300])
def test_matches_brute_force(data, k):
    reference, queries = data
    values, indices = TorchNearestNeighbors().fit(reference).search(queries, k)
    expected_values, expected_indices = _reference_search(reference, queries, k)
    assert values.shape == indices.shape == (40, k)
    torch.testing.assert_close(values, expected_values)
    # random Gaussian data: no ties, so the neighbours themselves must match
    assert torch.equal(indices, expected_indices)


def test_values_ascending_and_non_negative(data):
    reference, queries = data
    # include exact duplicates, where cancellation could make the distance slightly negative
    values, _ = (
        TorchNearestNeighbors().fit(reference).search(torch.cat([queries, reference[:5]]), 10)
    )
    assert (values >= 0).all()
    assert (values[:, 1:] >= values[:, :-1]).all()


@pytest.mark.parametrize("batch_size", [1, 7, None, "auto", 10_000])
def test_batch_size_does_not_change_results(data, batch_size):
    reference, queries = data
    expected = TorchNearestNeighbors(batch_size=None).fit(reference).search(queries, 5)
    result = TorchNearestNeighbors(batch_size=batch_size).fit(reference).search(queries, 5)
    torch.testing.assert_close(result[0], expected[0])
    assert torch.equal(result[1], expected[1])


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, "x", True])
def test_invalid_batch_size_raises(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        TorchNearestNeighbors(batch_size=batch_size)


def test_float32_results_stay_float32(data):
    reference, queries = data
    values, indices = TorchNearestNeighbors().fit(reference.float()).search(queries.float(), 5)
    assert values.dtype == torch.float32
    assert indices.dtype == torch.long
    # float32 cancellation in ||q||^2 + ||r||^2 - 2 q.r: relative errors around 1e-5 here
    torch.testing.assert_close(
        values.double(), _reference_search(reference, queries, 5)[0], rtol=1e-4, atol=1e-4
    )


def test_results_follow_query_dtype(data):
    reference, queries = data
    values, _ = TorchNearestNeighbors().fit(reference).search(queries.float(), 5)
    assert values.dtype == torch.float32


def test_k_larger_than_reference_raises(data):
    reference, queries = data
    search = TorchNearestNeighbors().fit(reference[:10])
    with pytest.raises(ValueError, match="k=11"):
        search.search(queries, 11)


def test_search_before_fit_raises(data):
    with pytest.raises(RequiresFittingException):
        TorchNearestNeighbors().search(data[1], 3)


def test_unpinned_follows_to(data):
    search = TorchNearestNeighbors().fit(data[0])
    assert search.to("meta") is search
    assert search.reference.device.type == "meta"


def test_pinned_ignores_to(data):
    search = TorchNearestNeighbors(device="cpu").fit(data[0])
    search.to("meta")
    assert search.reference.device.type == "cpu"


def test_pinned_moves_reference_on_fit(data):
    search = TorchNearestNeighbors(device="meta").fit(data[0])
    assert search.reference.device.type == "meta"


def test_unpinned_keeps_reference_where_it_was_fitted(data):
    search = TorchNearestNeighbors().fit(data[0].to("meta"))
    assert search.reference.device.type == "meta"


@requires(cuda=1)
def test_pinned_cpu_with_accelerator_queries(data, accelerator):
    reference, queries = data
    search = TorchNearestNeighbors(device="cpu").fit(reference.to(accelerator))
    assert search.reference.device.type == "cpu"
    values, indices = search.search(queries.to(accelerator), 5)
    assert values.device == indices.device == accelerator
    expected_values, expected_indices = _reference_search(reference, queries, 5)
    torch.testing.assert_close(values.cpu(), expected_values)
    assert torch.equal(indices.cpu(), expected_indices)


@requires(cuda=1)
def test_unpinned_on_accelerator(data, accelerator):
    reference, queries = data
    search = TorchNearestNeighbors(batch_size="auto").fit(reference).to(accelerator)
    assert search.reference.device == accelerator
    values, _ = search.search(queries, 5)
    assert values.device == queries.device
    torch.testing.assert_close(values, _reference_search(reference, queries, 5)[0])


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(data, k):
    search = TorchNearestNeighbors().fit(data[0])
    with pytest.raises(ValueError, match="k must be at least 1"):
        search.search(data[1], k)


@pytest.fixture
def block_rows(monkeypatch):
    """Records the number of queries in each distance block. Results do not depend on the block
    size, so this is the only way to see it; it relies on the search calling ``torch.addmm``
    once per block."""
    rows = []
    addmm = torch.addmm

    def recording_addmm(input, mat1, mat2, **kwargs):
        rows.append(mat1.shape[0])
        return addmm(input, mat1, mat2, **kwargs)

    monkeypatch.setattr(torch, "addmm", recording_addmm)
    return rows


@pytest.mark.parametrize(
    "batch_size, expected", [(7, [7, 7, 7, 7, 7, 5]), (40, [40]), (100, [40]), (None, [40])]
)
def test_blocks_follow_batch_size(data, block_rows, batch_size, expected):
    reference, queries = data
    TorchNearestNeighbors(batch_size=batch_size).fit(reference).search(queries, 3)
    assert block_rows == expected


@requires(cuda=1)
def test_auto_batch_size_follows_free_gpu_memory(data, block_rows, monkeypatch, accelerator):
    reference, queries = data
    n, itemsize = reference.shape[0], reference.element_size()
    calls = []

    def mem_get_info(device=None):
        calls.append(device)
        # half of the free memory is the budget, for two distance blocks of 16 rows
        return 2 * (2 * 16 * n * itemsize), 1 << 40

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    search = TorchNearestNeighbors().fit(reference.to(accelerator))
    search.search(queries.to(accelerator), 3)
    assert calls
    assert block_rows == [16, 16, 8]


def test_auto_batch_size_follows_memory_budget(data, block_rows, monkeypatch):
    from pytorch_ood.utils import neighbors

    reference, queries = data
    n, itemsize = reference.shape[0], reference.element_size()
    # budget for two distance blocks of 16 rows
    monkeypatch.setattr(neighbors, "_CPU_BLOCK_BYTES", 2 * 16 * n * itemsize)
    TorchNearestNeighbors().fit(reference).search(queries, 3)
    assert block_rows == [16, 16, 8]

    # a budget below one row still processes one row at a time
    block_rows.clear()
    monkeypatch.setattr(neighbors, "_CPU_BLOCK_BYTES", 1)
    TorchNearestNeighbors().fit(reference).search(queries[:3], 3)
    assert block_rows == [1, 1, 1]


@pytest.mark.parametrize("batch_size", ["auto", None, 3])
def test_empty_queries(data, batch_size):
    search = TorchNearestNeighbors(batch_size=batch_size).fit(data[0])
    values, indices = search.search(data[1][:0].float(), 5)
    assert values.shape == indices.shape == (0, 5)
    assert values.dtype == torch.float32
    assert indices.dtype == torch.long


def test_none_is_one_block_for_many_queries(block_rows, seed):
    reference = torch.randn(8, 4, generator=seed)
    queries = torch.randn(5000, 4, generator=seed)
    TorchNearestNeighbors(batch_size=None).fit(reference).search(queries, 3)
    assert block_rows == [5000]


@pytest.mark.parametrize("shape", [(5, 15), (5,), (2, 5, 16)])
def test_queries_must_match_reference_dimension(data, shape):
    search = TorchNearestNeighbors().fit(data[0])
    with pytest.raises(ValueError, match="B x 16"):
        search.search(torch.randn(shape, dtype=torch.float64), 3)


def test_inputs_requiring_grad(data):
    reference, queries = (t.clone().requires_grad_() for t in data)
    values, _ = TorchNearestNeighbors().fit(reference).search(queries, 3)
    assert not values.requires_grad


@pytest.mark.parametrize("cls", [NearestNeighborSearch, TorchNearestNeighbors])
def test_docstring_examples_run(cls):
    blocks = list(code_blocks(cls.__doc__))
    assert blocks
    for code in blocks:
        exec(compile(code, f"<{cls.__name__} example>", "exec"), {})
