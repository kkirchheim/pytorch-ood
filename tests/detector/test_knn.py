import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import KNN
from tests.helpers import ClassificationModel, sample_dataset


def _kth_distance(z_train, z_test, k, normalize=True):
    """Distance to the k-th nearest training feature, after L2 normalization (paper, Alg. 1)."""
    if normalize:
        z_train = z_train / z_train.norm(dim=1, keepdim=True)
        z_test = z_test / z_test.norm(dim=1, keepdim=True)
    return torch.cdist(z_test.double(), z_train.double()).sort(dim=1).values[:, k - 1]


class TestKNN(unittest.TestCase):
    """
    Tests for k-Nearest Neighbor
    """

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

    def test_k_checked_against_id_samples_at_fit(self):
        """OOD samples do not count towards k when fitting."""
        labels = torch.cat([torch.zeros(10, dtype=torch.long), -torch.ones(10, dtype=torch.long)])
        with self.assertRaisesRegex(ValueError, r"k=15 exceeds the number of ID samples \(10\)"):
            KNN(None, k=15).fit_features(torch.randn(20, 4), labels)

    def test_zero_vectors_give_finite_scores(self):
        z_train = torch.randn(20, 4)
        z_train[3] = 0
        detector = KNN(None, k=3).fit_features(z_train, torch.zeros(20, dtype=torch.long))
        z_test = torch.cat([torch.zeros(1, 4), torch.randn(4, 4)])
        self.assertTrue(torch.isfinite(detector.predict_features(z_test)).all())

    def test_fit_features_moves_unpinned_reference_to_detector_device(self):
        detector = KNN(torch.nn.Linear(4, 4).to("meta"), k=3)
        detector.fit_features(torch.randn(20, 4), torch.zeros(20, dtype=torch.long))
        self.assertEqual(detector.search.reference.device.type, "meta")

    def test_sequential_encoder(self):
        """An encoder that evaluates to False, like an empty Sequential, is still an encoder."""
        detector = KNN(torch.nn.Sequential(), k=3)
        detector.fit(
            DataLoader(TensorDataset(torch.randn(20, 4), torch.zeros(20, dtype=torch.long)))
        )
        self.assertEqual(detector.predict(torch.randn(5, 4)).shape, (5,))
