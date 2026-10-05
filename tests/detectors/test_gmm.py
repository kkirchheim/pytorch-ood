import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import ModelNotSetException, RequiresFittingException
from src.pytorch_ood.detector import GMM
from tests.helpers import ClassificationModel


def _classes(sizes=(100, 400, 300, 50), scales=(0.3, 1.0, 3.0, 0.5), dim=8, seed=0):
    """Gaussian classes with different sizes and spreads, so weights and determinants matter."""
    g = torch.Generator().manual_seed(seed)
    z = torch.cat(
        [
            torch.randn(n, dim, generator=g) * s + 4 * torch.eye(len(sizes), dim)[k]
            for k, (n, s) in enumerate(zip(sizes, scales))
        ]
    )
    y = torch.cat([torch.full((n,), k) for k, n in enumerate(sizes)])
    return z, y


def _mixture_nll(z_fit, y_fit, z, reg):
    """-log sum_k pi_k N(z | mu_k, Sigma_k), with torch.distributions as reference."""
    z_fit, z = z_fit.double(), z.double()
    log_probs = []
    for k in y_fit.unique():
        z_k = z_fit[y_fit == k]
        mu = z_k.mean(dim=0)
        cov = (z_k - mu).T @ (z_k - mu) / len(z_k) + reg * torch.eye(z.shape[1], dtype=z.dtype)
        normal = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        log_probs.append(normal.log_prob(z) + torch.tensor(len(z_k) / len(z_fit)).log())
    return -torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)


class GMMTest(unittest.TestCase):
    """
    Test GMM detector
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_fit_predict(self):
        nn = ClassificationModel()
        detector = GMM(nn)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader)
        scores = detector(x)

        self.assertEqual(scores.shape, (20,))
        self.assertIsNotNone(scores)

    def test_fit_predict_features(self):
        detector = GMM(encoder=None)

        z = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])

        detector.fit_features(z, y)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (20,))

    def test_nofit(self):
        nn = ClassificationModel()
        detector = GMM(nn)
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            detector(x)

    def test_no_model(self):
        detector = GMM(encoder=None)

        z = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        detector.fit_features(z, y)

        with self.assertRaises(ModelNotSetException):
            detector(torch.randn(size=(5, 10)))

    def test_matches_mixture_nll(self):
        z, y = _classes()
        test = torch.randn(64, 8) * 2 + 1
        detector = GMM(None, reg=1e-3).fit_features(z, y)
        torch.testing.assert_close(
            detector.predict_features(test).double(), _mixture_nll(z, y, test, reg=1e-3)
        )

    def test_mixing_weights_matter(self):
        """Only the class sizes differ, which changes the mixture."""
        z, y = _classes(sizes=(100, 100, 100, 100))
        test = torch.randn(32, 8) * 2
        balanced = GMM(None).fit_features(z, y).predict_features(test)
        # duplicate the samples of class 0, which keeps its mean and covariance
        z_dup = torch.cat([z, z[y == 0], z[y == 0]])
        y_dup = torch.cat([y, y[y == 0], y[y == 0]])
        weighted = GMM(None).fit_features(z_dup, y_dup).predict_features(test)
        self.assertFalse(torch.allclose(balanced, weighted))
        torch.testing.assert_close(weighted.double(), _mixture_nll(z_dup, y_dup, test, reg=1e-6))

    def test_ignores_ood_samples(self):
        z, y = _classes()
        z_ood = torch.randn(50, 8) * 10
        test = torch.randn(16, 8)
        detector = GMM(None).fit_features(torch.cat([z, z_ood]), torch.cat([y, -torch.ones(50)]))
        torch.testing.assert_close(
            detector.predict_features(test), GMM(None).fit_features(z, y).predict_features(test)
        )

    def test_fewer_samples_than_dimensions(self):
        """Singular class covariances are regularized and give finite scores."""
        z, y = _classes(sizes=(5, 5), scales=(1.0, 1.0), dim=32)
        scores = GMM(None).fit_features(z, y).predict_features(torch.randn(10, 32))
        self.assertTrue(torch.isfinite(scores).all())
