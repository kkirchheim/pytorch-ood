"""How the detectors built on a ``NearestNeighborSearch`` (KNN, NNGuide) use the search."""

import pytest
import torch

from pytorch_ood.detector import KNN, NNGuide
from pytorch_ood.utils import NearestNeighborSearch, TorchNearestNeighbors
from tests.helpers import requires


class _BruteForceSearch(NearestNeighborSearch):
    """A minimal third-party search; keeps the base class's no-op ``to``."""

    def __init__(self, metric):
        self.metric = metric
        self.calls = 0
        self.reference = None

    def fit(self, reference):
        self.reference = reference
        return self

    def search(self, queries, k):
        self.calls += 1
        if self.metric == "sqeuclidean":
            return (torch.cdist(queries, self.reference) ** 2).topk(k, dim=1, largest=False)
        return (queries @ self.reference.T).topk(k, dim=1)

    @property
    def n_samples(self):
        return 0 if self.reference is None else len(self.reference)


@pytest.fixture(params=["KNN", "NNGuide"])
def case(request):
    """A factory for the detector and its metric. The detectors of one test share their modules,
    which also give them a device."""
    if request.param == "KNN":
        encoder = torch.nn.Linear(4, 4)
        return (lambda **kwargs: KNN(encoder, **kwargs)), "sqeuclidean"
    head = torch.nn.Linear(4, 3)
    return (lambda **kwargs: NNGuide(None, head, **kwargs)), "inner_product"


@pytest.fixture
def data(seed):
    z = torch.randn(40, 4, generator=seed)
    return z, torch.zeros(40, dtype=torch.long), torch.randn(8, 4, generator=seed)


def test_default_search(case):
    make, metric = case
    search = make().search
    assert isinstance(search, TorchNearestNeighbors)
    assert search.metric == metric


def test_custom_search(case, data):
    make, metric = case
    z, labels, queries = data
    search = _BruteForceSearch(metric)
    detector = make(k=4, search=search).fit_features(z, labels)
    expected = make(k=4).fit_features(z, labels).predict_features(queries)
    torch.testing.assert_close(detector.predict_features(queries), expected)
    assert search.calls == 1
    # the base class's ``to`` is a no-op
    detector.to("cpu")
    assert detector.search is search


@pytest.mark.parametrize("search", [object(), TorchNearestNeighbors], ids=["object", "class"])
def test_rejects_object_that_is_not_a_search(case, search):
    with pytest.raises(TypeError, match="NearestNeighborSearch"):
        case[0](search=search)


def test_rejects_search_with_other_metric(case):
    make, metric = case
    other = "inner_product" if metric == "sqeuclidean" else "sqeuclidean"
    with pytest.raises(ValueError, match=f"needs a search with metric '{metric}'"):
        make(search=TorchNearestNeighbors(metric=other))


@pytest.mark.parametrize(
    "k, error, match",
    [(2.5, TypeError, "integer"), (True, TypeError, "integer"), (0, ValueError, "at least 1")],
)
def test_invalid_k_raises(case, data, k, error, match):
    make = case[0]
    with pytest.raises(error, match=match):
        make(k=k)
    # k set after construction, e.g. by GridSearch, is checked when fitting
    detector = make(k=3)
    detector.k = k
    with pytest.raises(error, match=match):
        detector.fit_features(*data[:2])


def test_k_can_change_after_fit(case, data):
    """k is a query-time parameter, so GridSearch can sweep it without refitting."""
    make = case[0]
    z, labels, queries = data
    detector = make(k=3).fit_features(z, labels)
    detector.k = 9
    expected = make(k=9).fit_features(z, labels).predict_features(queries)
    torch.testing.assert_close(detector.predict_features(queries), expected)


def test_k_checked_against_the_search(case, data):
    """k is checked against the search's size, without calling it; OOD samples do not count."""
    make, metric = case
    z, _, queries = data
    search = _BruteForceSearch(metric)
    labels = torch.cat([torch.zeros(10, dtype=torch.long), -torch.ones(30, dtype=torch.long)])
    detector = make(k=3, search=search).fit_features(z, labels)
    detector.k = 11
    with pytest.raises(ValueError, match=r"k=11 exceeds the number of fitted samples \(10\)"):
        detector.predict_features(queries)
    assert search.calls == 0
    # the size comes from the search, not from a copy in the detector
    search.fit(z[:20])
    assert detector.predict_features(queries).shape == (8,)


def test_to_calls_search_to_in_place(case):
    """``to`` calls the search's ``to`` and keeps the search, whatever ``to`` returns."""
    make, metric = case

    class InPlaceSearch(_BruteForceSearch):
        def to(self, device):
            self.moved_to = device  # returns None, like an in-place move

    search = InPlaceSearch(metric)
    detector = make(search=search)
    assert detector.to("meta") is detector
    assert detector.search is search
    assert torch.device(search.moved_to).type == "meta"


@pytest.mark.parametrize("pinned", [False, True])
def test_to_moves_unpinned_search(case, data, pinned):
    make, metric = case
    search = TorchNearestNeighbors(device="cpu" if pinned else None, metric=metric)
    detector = make(k=3, search=search).fit_features(*data[:2])
    detector.to("meta")
    assert detector.search.reference.device.type == ("cpu" if pinned else "meta")


@requires(cuda=1)
def test_on_accelerator_matches_cpu(case, data, accelerator):
    make = case[0]
    z, labels, queries = data
    expected = make(k=5).fit_features(z, labels).predict_features(queries)
    # CPU features into a detector on the accelerator: the search follows the detector
    detector = make(k=5).to(accelerator).fit_features(z, labels)
    assert detector.search.reference.device == accelerator
    scores = detector.predict_features(queries)
    assert scores.device == accelerator
    torch.testing.assert_close(scores.cpu(), expected)
