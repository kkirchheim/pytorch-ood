import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import Mahalanobis
from tests.helpers import ClassificationModel


class MahalanobisTest(unittest.TestCase):
    """
    Test mahalanobis method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_something(self):
        nn = ClassificationModel()
        model = Mahalanobis(nn)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(scores)

        scores = model(torch.ones(size=(10, 10)) * 10)
        print(scores)

        self.assertIsNotNone(scores)

    def test_nofit(self):
        nn = ClassificationModel()
        model = Mahalanobis(nn)
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            model(x)


class MahalanobisScoreTest(unittest.TestCase):
    def test_half_squared_distance_to_closest_center(self):
        detector = Mahalanobis(None)
        detector.mu = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
        detector.precision = torch.diag(torch.tensor([1.0, 4.0]))
        x = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
        # squared distances: (5, 13) and (9, 1)
        torch.testing.assert_close(detector.predict_features(x), torch.tensor([2.5, 0.5]))


class MahalanobisEpsTest(unittest.TestCase):
    """
    Tests for the configurable eps parameter added to the Mahalanobis detector.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_default_eps(self):
        """Mahalanobis should use eps=1e-6 by default."""
        model = Mahalanobis(None)
        self.assertEqual(model.eps, 1e-6)

    def test_custom_eps(self):
        """Mahalanobis should accept a custom eps value."""
        model = Mahalanobis(None, eps=1e-3)
        self.assertEqual(model.eps, 1e-3)

    def test_eps_applied_to_covariance(self):
        """The eps value should be added to the diagonal of the covariance matrix during fit."""
        x = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))]).long()

        model_a = Mahalanobis(None, eps=0.0).fit_features(x, y)
        model_b = Mahalanobis(None, eps=1e-2).fit_features(x, y)

        torch.testing.assert_close(model_b.cov - model_a.cov, torch.eye(10) * 1e-2)
        torch.testing.assert_close(model_b.precision, torch.linalg.inv(model_b.cov))

    def test_eps_makes_singular_covariance_invertible(self):
        """
        eps=0 on near-singular data can lead to singular matrices.
        With eps > 0 the same data should produce a finite precision matrix.
        """
        nn = ClassificationModel()
        # Use identical samples to create a (near-)singular covariance
        x_const = torch.ones(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        dataset = TensorDataset(x_const, y)
        loader = DataLoader(dataset)

        model_safe = Mahalanobis(nn, eps=1e-3)
        model_safe.fit(loader)
        self.assertTrue(
            torch.isfinite(model_safe.precision).all(),
            "With eps=1e-3, precision matrix should be finite even for constant inputs.",
        )
