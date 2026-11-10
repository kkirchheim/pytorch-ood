import pytest
import torch

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.utils import NearestNeighborSearch, TorchNearestNeighbors
from tests.helpers import requires


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
