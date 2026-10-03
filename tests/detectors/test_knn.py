import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import ModelNotSetException, RequiresFittingException
from src.pytorch_ood.detector import KNN
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

    def test_requires_fitting(self):
        model = ClassificationModel()
        detector = KNN(model)

        x = torch.zeros(size=(128, 10))

        with self.assertRaises(RequiresFittingException):
            detector(x)

    def test_input(self):
        """ """
        model = ClassificationModel()
        detector = KNN(model)

        ds = sample_dataset(n_dim=10)
        loader = DataLoader(ds)
        detector.fit(loader)

        x = torch.randn(size=(128, 10))
        scores = detector(x)

        print(scores)
        self.assertIsNotNone(scores)

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

    def test_knn_kwargs_no_longer_conflict(self):
        """Passing sklearn kwargs must not collide with a hardcoded n_neighbors."""
        model = ClassificationModel()
        detector = KNN(model, k=3, metric="euclidean")
        ds = sample_dataset(n_dim=10, seed=3)
        detector.fit(DataLoader(ds))
        scores = detector(torch.randn(size=(16, 10)))
        self.assertEqual(scores.shape, (16,))

    def test_matches_paper(self):
        g = torch.Generator().manual_seed(0)
        z_train = torch.randn(300, 16, generator=g) * torch.rand(300, 1, generator=g) * 10
        z_test = torch.randn(40, 16, generator=g) * 5
        labels = torch.zeros(300, dtype=torch.long)
        for k in (1, 7, 50):
            with self.subTest(k=k):
                scores = KNN(None, k=k).fit_features(z_train, labels).predict_features(z_test)
                torch.testing.assert_close(scores, _kth_distance(z_train, z_test, k))

    def test_invariant_to_feature_norm(self):
        """With normalization, only the direction of the features matters."""
        g = torch.Generator().manual_seed(1)
        z_train = torch.randn(200, 8, generator=g)
        z_test = torch.randn(30, 8, generator=g)
        labels = torch.zeros(200, dtype=torch.long)
        scale_train = torch.rand(200, 1, generator=g) * 100 + 0.01
        scale_test = torch.rand(30, 1, generator=g) * 100 + 0.01
        expected = KNN(None).fit_features(z_train, labels).predict_features(z_test)
        scaled = KNN(None).fit_features(z_train * scale_train, labels)
        torch.testing.assert_close(scaled.predict_features(z_test * scale_test), expected)

    def test_without_normalization(self):
        g = torch.Generator().manual_seed(2)
        z_train = torch.randn(200, 8, generator=g) * 3
        z_test = torch.randn(30, 8, generator=g)
        labels = torch.zeros(200, dtype=torch.long)
        detector = KNN(None, k=5, normalize=False).fit_features(z_train, labels)
        torch.testing.assert_close(
            detector.predict_features(z_test), _kth_distance(z_train, z_test, 5, normalize=False)
        )

    def test_ignores_ood_samples(self):
        g = torch.Generator().manual_seed(3)
        z_train = torch.randn(100, 8, generator=g)
        z_ood = torch.randn(50, 8, generator=g) * 0.01
        z_test = torch.randn(20, 8, generator=g)
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
