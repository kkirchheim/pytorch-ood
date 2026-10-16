import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import Residual, ViM
from tests.helpers import ClassificationModel


def _reference_vim(features, train_features, w, b, d):
    """
    Reference ViM implementation (as before ViM was refactored to extend Residual),
    used to ensure the refactoring does not change ViM's scores.
    """
    u = -(torch.linalg.pinv(w) @ b)
    X = train_features - u
    cov = (X.T @ X) / X.shape[0]
    _, eig_vecs = torch.linalg.eigh(cov)
    ns = eig_vecs[:, : eig_vecs.shape[1] - d]
    alpha = (train_features @ w.T + b).max(dim=-1).values.mean() / (X @ ns).norm(dim=-1).mean()
    vlogit = ((features - u) @ ns).norm(dim=-1) * alpha
    energy = torch.logsumexp(features @ w.T + b, dim=-1)
    return vlogit - energy


class TestResidual(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.n_dim = 10
        self.w = torch.randn(size=(3, self.n_dim))
        self.b = torch.randn(size=(3,))
        self.train = torch.randn(size=(200, self.n_dim))
        self.labels = torch.randint(0, 3, size=(200,))
        self.test = torch.randn(size=(64, self.n_dim))

    def test_fit_predict(self):
        model = ClassificationModel(n_hidden=10)
        detector = Residual(
            model.features, d=2, w=model.classifier.weight.data, b=model.classifier.bias.data
        )
        loader = DataLoader(TensorDataset(torch.randn(size=(20, 10)), torch.zeros(20)))
        detector.fit(loader)

        scores = detector(torch.zeros(size=(128, 10)))
        self.assertEqual(scores.shape, (128,))
        self.assertTrue((scores >= 0).all())

    def test_requires_fitting(self):
        detector = Residual(None, d=2, w=self.w, b=self.b)
        with self.assertRaises(RequiresFittingException):
            detector.predict_features(self.test)

    def test_principal_subspace_has_zero_residual(self):
        """
        Features that lie entirely in the principal subspace (after centering) have a residual norm of zero.
        """
        detector = Residual(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        x = self.test - detector.u
        x = x - (x @ detector.principal_subspace) @ detector.principal_subspace.T + detector.u
        torch.testing.assert_close(
            detector.predict_features(x), torch.zeros(64), atol=1e-4, rtol=0
        )

    def test_vim_scores_unchanged(self):
        vim = ViM(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        expected = _reference_vim(self.test, self.train, self.w, self.b, d=4)
        torch.testing.assert_close(vim.predict_features(self.test), expected)

    def test_vim_extends_residual(self):
        residual = Residual(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)
        vim = ViM(None, d=4, w=self.w, b=self.b).fit_features(self.train, self.labels)

        self.assertIsInstance(vim, Residual)
        energy = torch.logsumexp(self.test @ self.w.T + self.b, dim=-1)
        expected = vim.alpha * residual.predict_features(self.test) - energy
        torch.testing.assert_close(vim.predict_features(self.test), expected)

    def test_repr(self):
        self.assertEqual(repr(Residual(None, d=4, w=self.w, b=self.b)), "Residual(d=4)")
        self.assertEqual(repr(ViM(None, d=4, w=self.w, b=self.b)), "ViM(d=4)")
