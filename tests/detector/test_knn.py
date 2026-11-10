import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import ModelNotSetException, RequiresFittingException
from pytorch_ood.detector import KNN
from pytorch_ood.utils import NearestNeighborSearch, TorchNearestNeighbors
from tests.helpers import ClassificationModel, sample_dataset


def _kth_distance(z_train, z_test, k, normalize=True):
    """Distance to the k-th nearest training feature, after L2 normalization (paper, Alg. 1)."""
    if normalize:
        z_train = z_train / z_train.norm(dim=1, keepdim=True)
        z_test = z_test / z_test.norm(dim=1, keepdim=True)
    return torch.cdist(z_test.double(), z_train.double()).sort(dim=1).values[:, k - 1]


class _SortingSearch(NearestNeighborSearch):
    """A minimal third-party backend; keeps the base class's no-op ``to``."""

    metric = "sqeuclidean"

    def __init__(self):
        self.calls = 0
        self.reference = None

    def fit(self, reference):
        self.reference = reference
        return self

    def search(self, queries, k):
        self.calls += 1
        return (torch.cdist(queries, self.reference) ** 2).topk(k, dim=1, largest=False)

    @property
    def n_samples(self):
        return 0 if self.reference is None else len(self.reference)


class TestKNN(unittest.TestCase):
    """
    Tests for k-Nearest Neighbor
    """

    def test_requires_fitting(self):
        model = ClassificationModel()
        detector = KNN(model)

        x = torch.zeros(size=(128, 10))

        with self.assertRaises(RequiresFittingException):
            detector(x)

    def test_input(self):
        """``detector(x)`` scores the encoder's features of ``x``."""
        model = ClassificationModel().eval()  # no dropout, so both forward passes agree
        detector = KNN(model)
        detector.fit(DataLoader(sample_dataset(n_dim=10)))

        x = torch.randn(size=(128, 10))
        scores = detector(x)

        self.assertEqual(scores.shape, (128,))
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(scores, detector.predict_features(model(x)))

    def test_k_neighbors(self):
        """The k-th nearest neighbor distance is non-decreasing in k."""
        g = torch.Generator().manual_seed(1)
        z_train = torch.randn(size=(200, 10), generator=g)
        y_train = torch.zeros(200, dtype=torch.long)
        z_test = torch.randn(size=(64, 10), generator=g)

        d1 = KNN(None, k=1).fit_features(z_train, y_train)
        d5 = KNN(None, k=5).fit_features(z_train, y_train)

        s1 = d1.predict_features(z_test)
        s5 = d5.predict_features(z_test)

        self.assertEqual(s1.shape, (64,))
        self.assertEqual(s5.shape, (64,))
        # distance to the 5th neighbor is at least the distance to the 1st
        self.assertTrue(torch.all(s5 >= s1 - 1e-6))

    def test_unknown_kwargs_raise(self):
        """There are no backend kwargs on the detector; backends are configured via ``search``."""
        with self.assertRaises(TypeError):
            KNN(None, k=3, metric="euclidean")

    def test_matches_paper(self):
        g = torch.Generator().manual_seed(0)
        dtype = torch.float64
        z_train = torch.randn(300, 16, generator=g, dtype=dtype)
        z_train = z_train * torch.rand(300, 1, generator=g, dtype=dtype) * 10
        z_test = torch.randn(40, 16, generator=g, dtype=dtype) * 5
        labels = torch.zeros(300, dtype=torch.long)
        for k in (1, 7, 50):
            with self.subTest(k=k):
                scores = KNN(None, k=k).fit_features(z_train, labels).predict_features(z_test)
                torch.testing.assert_close(scores, _kth_distance(z_train, z_test, k))

    def test_float32_features_give_float32_scores(self):
        g = torch.Generator().manual_seed(0)
        z_train = torch.randn(300, 16, generator=g)
        z_test = torch.randn(40, 16, generator=g)
        labels = torch.zeros(300, dtype=torch.long)
        scores = KNN(None, k=7).fit_features(z_train, labels).predict_features(z_test)
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(
            scores.double(), _kth_distance(z_train, z_test, 7), rtol=1e-4, atol=1e-5
        )

    def test_invariant_to_feature_norm(self):
        """With normalization, only the direction of the features matters."""
        g = torch.Generator().manual_seed(1)
        dtype = torch.float64
        z_train = torch.randn(200, 8, generator=g, dtype=dtype)
        z_test = torch.randn(30, 8, generator=g, dtype=dtype)
        labels = torch.zeros(200, dtype=torch.long)
        scale_train = torch.rand(200, 1, generator=g, dtype=dtype) * 100 + 0.01
        scale_test = torch.rand(30, 1, generator=g, dtype=dtype) * 100 + 0.01
        expected = KNN(None).fit_features(z_train, labels).predict_features(z_test)
        scaled = KNN(None).fit_features(z_train * scale_train, labels)
        torch.testing.assert_close(scaled.predict_features(z_test * scale_test), expected)

    def test_without_normalization(self):
        g = torch.Generator().manual_seed(2)
        z_train = torch.randn(200, 8, generator=g, dtype=torch.float64) * 3
        z_test = torch.randn(30, 8, generator=g, dtype=torch.float64)
        labels = torch.zeros(200, dtype=torch.long)
        detector = KNN(None, k=5, normalize=False).fit_features(z_train, labels)
        torch.testing.assert_close(
            detector.predict_features(z_test), _kth_distance(z_train, z_test, 5, normalize=False)
        )

    def test_ignores_ood_samples(self):
        g = torch.Generator().manual_seed(3)
        z_train = torch.randn(100, 8, generator=g, dtype=torch.float64)
        z_ood = torch.randn(50, 8, generator=g, dtype=torch.float64) * 0.01
        z_test = torch.randn(20, 8, generator=g, dtype=torch.float64)
        labels = torch.cat([torch.zeros(100, dtype=torch.long), -torch.ones(50, dtype=torch.long)])
        detector = KNN(None, k=3).fit_features(torch.cat([z_train, z_ood]), labels)
        torch.testing.assert_close(
            detector.predict_features(z_test), _kth_distance(z_train, z_test, 3)
        )

    def test_defaults(self):
        detector = KNN(None)
        self.assertEqual(detector.k, 50)
        self.assertTrue(detector.normalize)

    def test_k_larger_than_fit_data_raises(self):
        with self.assertRaisesRegex(ValueError, "k=50 exceeds"):
            KNN(None).fit_features(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))
        # boundary: k = number of ID samples is fine, one more is not
        KNN(None, k=10).fit_features(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, r"k=11 exceeds the number of ID samples \(10\)"):
            KNN(None, k=11).fit_features(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))

    def test_no_id_samples_raises(self):
        with self.assertRaisesRegex(ValueError, "No ID samples"):
            KNN(None, k=1).fit_features(torch.randn(10, 4), -torch.ones(10, dtype=torch.long))

    def test_non_integer_k_raises(self):
        with self.assertRaisesRegex(TypeError, "k must be an integer"):
            KNN(None, k=2.5)

    def test_k_can_change_after_fit(self):
        """k is a query-time parameter, so GridSearch can sweep it without refitting."""
        g = torch.Generator().manual_seed(4)
        z_train = torch.randn(100, 8, generator=g, dtype=torch.float64)
        z_test = torch.randn(20, 8, generator=g, dtype=torch.float64)
        detector = KNN(None, k=3).fit_features(z_train, torch.zeros(100, dtype=torch.long))
        detector.k = 9
        torch.testing.assert_close(
            detector.predict_features(z_test), _kth_distance(z_train, z_test, 9)
        )
        detector.k = 101
        with self.assertRaisesRegex(ValueError, "k=101 exceeds"):
            detector.predict_features(z_test)

    def test_default_search_backend(self):
        self.assertIsInstance(KNN(None).search, TorchNearestNeighbors)

    def test_custom_search_backend(self):
        g = torch.Generator().manual_seed(5)
        z_train = torch.randn(100, 8, generator=g, dtype=torch.float64)
        z_test = torch.randn(20, 8, generator=g, dtype=torch.float64)
        search = _SortingSearch()
        detector = KNN(None, k=4, search=search).fit_features(
            z_train, torch.zeros(100, dtype=torch.long)
        )
        torch.testing.assert_close(
            detector.predict_features(z_test), _kth_distance(z_train, z_test, 4)
        )
        self.assertEqual(search.calls, 1)
        # the base class's ``to`` is a no-op
        detector.to("cpu")
        self.assertIs(detector.search, search)

    def test_k_checked_before_search(self):
        """The detector rejects a too large k itself, without calling the backend; OOD samples
        do not count as fitted samples."""
        search = _SortingSearch()
        z = torch.randn(20, 4)
        labels = torch.cat([torch.zeros(10, dtype=torch.long), -torch.ones(10, dtype=torch.long)])
        detector = KNN(None, k=3, search=search).fit_features(z, labels)
        detector.k = 11
        with self.assertRaisesRegex(
            ValueError, r"k=11 exceeds the number of fitted samples \(10\)"
        ):
            detector.predict_features(z)
        self.assertEqual(search.calls, 0)

    def test_k_checked_against_the_search(self):
        """The fitted size comes from the search, not from a copy in the detector."""
        search = _SortingSearch()
        detector = KNN(None, k=3, search=search)
        detector.fit_features(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))
        search.fit(torch.randn(20, 4))
        detector.k = 15
        self.assertEqual(detector.predict_features(torch.randn(5, 4)).shape, (5,))

    def test_k_below_one_raises(self):
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "k must be at least 1"):
                KNN(None, k=k)
        # k set after construction, e.g. by GridSearch, is checked when fitting
        detector = KNN(None, k=3)
        detector.k = 0
        with self.assertRaisesRegex(ValueError, "k must be at least 1"):
            detector.fit_features(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))

    def test_rejects_object_that_is_not_a_search(self):
        with self.assertRaisesRegex(TypeError, "NearestNeighborSearch"):
            KNN(None, search=object())
        # the class instead of an instance
        with self.assertRaisesRegex(TypeError, "NearestNeighborSearch"):
            KNN(None, search=TorchNearestNeighbors)

    def test_to_calls_search_to_in_place(self):
        """``KNN.to`` calls the backend's ``to`` and keeps the backend, whatever ``to`` returns."""

        class InPlaceSearch(_SortingSearch):
            def to(self, device):
                self.moved_to = device  # returns None, like an in-place move

        search = InPlaceSearch()
        detector = KNN(torch.nn.Linear(4, 4), k=3, search=search)
        self.assertIs(detector.to("meta"), detector)
        self.assertIs(detector.search, search)
        self.assertEqual(torch.device(search.moved_to).type, "meta")
        self.assertEqual(next(detector.encoder.parameters()).device.type, "meta")

    def test_rejects_search_with_other_metric(self):
        search = _SortingSearch()
        search.metric = "inner_product"
        with self.assertRaisesRegex(ValueError, "sqeuclidean"):
            KNN(None, search=search)

    def test_to_moves_unpinned_reference(self):
        detector = KNN(None, k=3).fit_features(
            torch.randn(20, 4), torch.zeros(20, dtype=torch.long)
        )
        detector.to("meta")
        self.assertEqual(detector.search.reference.device.type, "meta")

    def test_to_keeps_pinned_reference(self):
        detector = KNN(torch.nn.Linear(4, 4), k=3, search=TorchNearestNeighbors(device="cpu"))
        detector.fit_features(torch.randn(20, 4), torch.zeros(20, dtype=torch.long))
        detector.to("meta")
        self.assertEqual(next(detector.encoder.parameters()).device.type, "meta")
        self.assertEqual(detector.search.reference.device.type, "cpu")

    def test_fit_features_moves_unpinned_reference_to_detector_device(self):
        detector = KNN(torch.nn.Linear(4, 4).to("meta"), k=3)
        detector.fit_features(torch.randn(20, 4), torch.zeros(20, dtype=torch.long))
        self.assertEqual(detector.search.reference.device.type, "meta")

    def test_without_encoder_raises(self):
        loader = DataLoader(TensorDataset(torch.randn(60, 4), torch.zeros(60, dtype=torch.long)))
        detector = KNN(None)
        with self.assertRaises(ModelNotSetException):
            detector.fit(loader)
        detector.fit_features(torch.randn(60, 4), torch.zeros(60, dtype=torch.long))
        with self.assertRaises(ModelNotSetException):
            detector.predict(torch.randn(3, 4))

    def test_sequential_encoder(self):
        """An encoder that evaluates to False, like an empty Sequential, is still an encoder."""
        detector = KNN(torch.nn.Sequential(), k=3)
        detector.fit(
            DataLoader(TensorDataset(torch.randn(20, 4), torch.zeros(20, dtype=torch.long)))
        )
        self.assertEqual(detector.predict(torch.randn(5, 4)).shape, (5,))
