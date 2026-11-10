import unittest

import torch
from torch.utils.data import DataLoader

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.detector import NNGuide
from pytorch_ood.utils import NearestNeighborSearch, TorchNearestNeighbors
from tests.helpers import ClassificationModel, requires, sample_dataset
from tests.helpers.components import code_blocks


class TestNNGuide(unittest.TestCase):
    """
    Tests for NNGuide Detector
    """

    def _make_detector(self):
        model = ClassificationModel(num_inputs=10, n_hidden=10, num_outputs=3)
        model.eval()
        return NNGuide(model.features, model.classifier, k=5)

    def test_requires_fitting(self):
        detector = self._make_detector()
        self.assertTrue(detector.requires_fit)
        z = torch.randn(16, 10)
        with self.assertRaises(RequiresFittingException):
            detector.predict_features(z)

    def test_fit_and_predict_features(self):
        detector = self._make_detector()
        ds = sample_dataset(n_dim=10)
        loader = DataLoader(ds, batch_size=32)

        detector.fit(loader)

        z = torch.randn(16, 10)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fit_features_direct(self):
        detector = self._make_detector()
        features = torch.randn(100, 10)
        labels = torch.zeros(100, dtype=torch.long)

        detector.fit_features(features, labels)

        z = torch.randn(8, 10)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (8,))

    def test_ignores_ood_in_fit(self):
        detector = self._make_detector()
        features = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.full((50,), -1)])

        detector.fit_features(features, labels)

        # only 50 ID samples should be in the bank
        self.assertEqual(detector.search.n_samples, 50)


def _reference_scores(head, z_train, z, k):
    # the paper, written out: normalized training features scaled by their energy, mean of the
    # k largest inner products with the normalized input features, times the energy of the input
    def energy(features):
        return torch.logsumexp(head(features), dim=1)

    def normalize(features):
        return features / features.norm(dim=1, keepdim=True)

    bank = normalize(z_train) * energy(z_train)[:, None]
    scores = []
    for zi in z:
        similarities = sorted((normalize(zi[None]) @ bank.T)[0].tolist(), reverse=True)
        guidance = sum(similarities[:k]) / k
        scores.append(-guidance * energy(zi[None])[0])
    return torch.stack(scores)


class TestNNGuideReference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = torch.nn.Linear(6, 4)
        self.z_train = torch.randn(200, 6)
        self.z = torch.randn(16, 6)

    def test_matches_reference(self):
        for k in (1, 5, 10):
            with self.subTest(k=k):
                detector = NNGuide(encoder=None, head=self.head, k=k)
                detector.fit_features(self.z_train, torch.zeros(200, dtype=torch.long))
                with torch.no_grad():
                    expected = _reference_scores(self.head, self.z_train, self.z, k)
                torch.testing.assert_close(detector.predict_features(self.z), expected)

    def test_chunks_do_not_change_scores(self):
        labels = torch.zeros(200, dtype=torch.long)
        expected = NNGuide(encoder=None, head=self.head, k=10).fit_features(self.z_train, labels)
        search = TorchNearestNeighbors(metric="inner_product", batch_size=3)
        detector = NNGuide(encoder=None, head=self.head, k=10, search=search)
        detector.fit_features(self.z_train, labels, batch_size=7)
        torch.testing.assert_close(
            detector.predict_features(self.z), expected.predict_features(self.z)
        )

    def test_neighbors_weighted_by_energy(self):
        # two training samples with the direction of the input, one with a larger energy: with
        # k = 1, the guidance is the inner product with the high-energy sample, while the cosine
        # similarity of both is 1
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        z_train = torch.tensor([[1.0, 0.0], [4.0, 0.0], [0.0, 1.0]])
        detector = NNGuide(encoder=None, head=head, k=1)
        detector.fit_features(z_train, torch.zeros(3, dtype=torch.long))
        z = torch.tensor([[2.0, 0.0]])
        energy = lambda f: torch.logsumexp(head(f), dim=1)  # noqa: E731
        with torch.no_grad():
            expected = -energy(z_train[1:2]) * energy(z)
        torch.testing.assert_close(detector.predict_features(z), expected)

    def test_fewer_samples_than_k_raises(self):
        detector = NNGuide(encoder=None, head=self.head, k=10)
        labels = torch.cat([torch.zeros(5, dtype=torch.long), torch.full((20,), -1)])
        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(25, 6), labels)


class _ExhaustiveSearch(NearestNeighborSearch):
    """A minimal third-party backend; keeps the base class's no-op ``to``."""

    metric = "inner_product"

    def __init__(self):
        self.calls = 0
        self.reference = None

    def fit(self, reference):
        self.reference = reference
        return self

    def search(self, queries, k):
        self.calls += 1
        return (queries @ self.reference.T).topk(k, dim=1)

    @property
    def n_samples(self):
        return 0 if self.reference is None else len(self.reference)


class TestNNGuideSearch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = torch.nn.Linear(6, 4)
        self.z_train = torch.randn(50, 6)
        self.labels = torch.zeros(50, dtype=torch.long)
        self.z = torch.randn(8, 6)

    def test_default_search(self):
        search = NNGuide(None, self.head).search
        self.assertIsInstance(search, TorchNearestNeighbors)
        self.assertEqual(search.metric, "inner_product")

    def test_custom_search(self):
        search = _ExhaustiveSearch()
        detector = NNGuide(None, self.head, k=4, search=search)
        detector.fit_features(self.z_train, self.labels)
        with torch.no_grad():
            expected = _reference_scores(self.head, self.z_train, self.z, 4)
        torch.testing.assert_close(detector.predict_features(self.z), expected)
        self.assertEqual(search.calls, 1)
        # the base class's ``to`` is a no-op
        detector.to("cpu")
        self.assertIs(detector.search, search)

    def test_rejects_object_that_is_not_a_search(self):
        for search in (object(), TorchNearestNeighbors):
            with self.subTest(search=search):
                with self.assertRaisesRegex(TypeError, "NearestNeighborSearch"):
                    NNGuide(None, self.head, search=search)

    def test_rejects_search_with_other_metric(self):
        with self.assertRaisesRegex(ValueError, "inner_product"):
            NNGuide(None, self.head, search=TorchNearestNeighbors())

    def test_invalid_k_raises(self):
        for k in (2.5, True):
            with self.subTest(k=k), self.assertRaisesRegex(TypeError, "k must be an integer"):
                NNGuide(None, self.head, k=k)
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "k must be at least 1"):
                NNGuide(None, self.head, k=k)

    def test_k_can_change_after_fit(self):
        detector = NNGuide(None, self.head, k=3).fit_features(self.z_train, self.labels)
        detector.k = 9
        with torch.no_grad():
            expected = _reference_scores(self.head, self.z_train, self.z, 9)
        torch.testing.assert_close(detector.predict_features(self.z), expected)

    def test_k_checked_before_search(self):
        search = _ExhaustiveSearch()
        labels = torch.cat([torch.zeros(10, dtype=torch.long), -torch.ones(40, dtype=torch.long)])
        detector = NNGuide(None, self.head, k=3, search=search)
        detector.fit_features(self.z_train, labels)
        detector.k = 11
        with self.assertRaisesRegex(
            ValueError, r"k=11 exceeds the number of fitted samples \(10\)"
        ):
            detector.predict_features(self.z)
        self.assertEqual(search.calls, 0)

    def test_search_is_refit(self):
        detector = NNGuide(None, self.head, k=3).fit_features(self.z_train, self.labels)
        detector.fit_features(self.z_train[:20], self.labels[:20])
        self.assertEqual(detector.search.n_samples, 20)

    def test_to_moves_unpinned_bank(self):
        detector = NNGuide(None, self.head, k=3).fit_features(self.z_train, self.labels)
        self.assertIs(detector.to("meta"), detector)
        self.assertEqual(detector.search.reference.device.type, "meta")

    def test_to_keeps_pinned_bank(self):
        search = TorchNearestNeighbors(metric="inner_product", device="cpu")
        detector = NNGuide(None, self.head, k=3, search=search)
        detector.fit_features(self.z_train, self.labels)
        detector.to("meta")
        self.assertEqual(next(detector.head.parameters()).device.type, "meta")
        self.assertEqual(detector.search.reference.device.type, "cpu")

    def test_bank_is_float32(self):
        detector = NNGuide(None, self.head, k=3)
        detector.fit_features(self.z_train.double(), self.labels)
        self.assertEqual(detector.search.reference.dtype, torch.float32)

    def test_docstring_example_runs(self):
        blocks = list(code_blocks(NNGuide.__doc__))
        self.assertTrue(blocks)
        for code in blocks:
            exec(compile(code, "<NNGuide example>", "exec"), {})


def test_fit_and_predict_match_features_path(seed):
    """``fit(loader)`` and ``detector(x)`` score like ``fit_features`` and ``predict_features``
    on the encoder's features."""
    model = ClassificationModel(num_inputs=10, n_hidden=10, num_outputs=3).eval()
    x_train = torch.randn(60, 10, generator=seed)
    labels = torch.cat([torch.zeros(50, dtype=torch.long), -torch.ones(10, dtype=torch.long)])
    x = torch.randn(8, 10, generator=seed)
    loader = DataLoader(list(zip(x_train, labels)), batch_size=16)

    detector = NNGuide(model.features, model.classifier, k=5).fit(loader)
    with torch.no_grad():
        expected = (
            NNGuide(None, model.classifier, k=5)
            .fit_features(model.features(x_train), labels)
            .predict_features(model.features(x))
        )
    torch.testing.assert_close(detector(x), expected)


@requires(cuda=1)
def test_on_accelerator_matches_cpu(seed, accelerator):
    head = torch.nn.Linear(6, 4)
    z_train = torch.randn(200, 6, generator=seed)
    labels = torch.zeros(200, dtype=torch.long)
    z = torch.randn(16, 6, generator=seed)
    expected = NNGuide(None, head, k=10).fit_features(z_train, labels).predict_features(z)

    detector = NNGuide(None, head, k=10).to(accelerator)
    detector.fit_features(z_train, labels)
    assert detector.search.reference.device == accelerator
    scores = detector.predict_features(z)
    assert scores.device == accelerator
    torch.testing.assert_close(scores.cpu(), expected)
