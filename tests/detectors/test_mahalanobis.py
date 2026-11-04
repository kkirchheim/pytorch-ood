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
        """Mahalanobis should use eps=1e-10 by default."""
        model = Mahalanobis(None)
        self.assertEqual(model.eps, 1e-10)

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
        expected = torch.linalg.inv(model_b.cov.double()).float()
        torch.testing.assert_close(model_b.precision, expected)

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


class MahalanobisPinvTest(unittest.TestCase):
    """
    Tests for the use_pinv parameter of the Mahalanobis detector.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(size=(40, 10))
        self.y = torch.cat([torch.zeros(size=(20,)), torch.ones(size=(20,))]).long()

    def test_default_is_inv(self):
        self.assertFalse(Mahalanobis(None).use_pinv)

    def test_pinv_precision(self):
        model = Mahalanobis(None, use_pinv=True).fit_features(self.x, self.y)
        expected = torch.linalg.pinv(model.cov.double(), hermitian=True).float()
        torch.testing.assert_close(model.precision, expected)

    def test_pinv_ignores_eps(self):
        """With pinv, no ridge is added: directions without variance are discarded instead."""
        model_a = Mahalanobis(None, eps=0.0, use_pinv=True).fit_features(self.x, self.y)
        model_b = Mahalanobis(None, eps=1e-2, use_pinv=True).fit_features(self.x, self.y)
        torch.testing.assert_close(model_a.cov, model_b.cov)
        torch.testing.assert_close(model_a.precision, model_b.precision)

    def test_pinv_matches_inv(self):
        """For a well-conditioned covariance matrix, inv and pinv should give the same scores."""
        model_inv = Mahalanobis(None).fit_features(self.x, self.y)
        model_pinv = Mahalanobis(None, use_pinv=True).fit_features(self.x, self.y)
        x = torch.randn(size=(16, 10))
        torch.testing.assert_close(
            model_pinv.predict_features(x), model_inv.predict_features(x), rtol=1e-4, atol=1e-4
        )


def _reference_scores(z, y, x, eps, use_pinv):
    """Float64 reference: half the squared distance to the closest center under the shared
    within-class covariance (sum of centered outer products / N), like sklearn's EmpiricalCovariance."""
    z, x = z.double(), x.double()
    n_classes = int(y.max()) + 1
    mu = torch.stack([z[y == k].mean(dim=0) for k in range(n_classes)])
    centered = z - mu[y]
    cov = centered.T @ centered / len(z)
    if use_pinv:
        precision = torch.linalg.pinv(cov, hermitian=True)
    else:
        precision = torch.linalg.inv(cov + eps * torch.eye(z.shape[1], dtype=torch.float64))
    d = torch.stack([((x - m) @ precision * (x - m)).sum(dim=1) for m in mu], dim=1)
    return 0.5 * d.min(dim=1).values


class MahalanobisNumericsTest(unittest.TestCase):
    """
    Rank-deficient covariance (fewer samples than feature dimensions): float32 inputs must give
    the scores of a float64 computation.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        n, d = 60, 80
        # decaying feature scales make the covariance ill-conditioned, like penultimate CNN features
        scales = torch.logspace(0, -3, d)
        self.y = torch.arange(n) % 3
        self.z = (torch.randn(n, d) * scales + torch.randn(3, d)[self.y]).relu()
        self.x = (torch.randn(16, d) * scales + torch.randn(3, d)[torch.arange(16) % 3]).relu()

    def _check(self, **kwargs):
        detector = Mahalanobis(None, **kwargs).fit_features(self.z, self.y)
        scores = detector.predict_features(self.x)
        self.assertEqual(scores.dtype, torch.float32)
        expected = _reference_scores(
            self.z, self.y, self.x, eps=detector.eps, use_pinv=detector.use_pinv
        )
        torch.testing.assert_close(scores.double(), expected, rtol=1e-3, atol=1e-3)

    def test_inv_matches_float64_reference(self):
        self._check()

    def test_pinv_matches_float64_reference(self):
        self._check(use_pinv=True)

    def test_float64_input_stays_float64(self):
        detector = Mahalanobis(None).fit_features(self.z.double(), self.y)
        self.assertEqual(detector.mu.dtype, torch.float64)
        self.assertEqual(detector.precision.dtype, torch.float64)
        self.assertEqual(detector.predict_features(self.x.double()).dtype, torch.float64)

    def test_float32_input_gives_float32_parameters(self):
        detector = Mahalanobis(None).fit_features(self.z, self.y)
        self.assertEqual(detector.mu.dtype, torch.float32)
        self.assertEqual(detector.precision.dtype, torch.float32)
