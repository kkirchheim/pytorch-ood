import unittest

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from pytorch_ood.detector import ViM
from pytorch_ood.loss import CrossEntropyLoss
from tests.helpers import ClassificationModel, sample_dataset


class TestViM(unittest.TestCase):
    """
    Tests for ViM
    """

    def test_logits(self):
        """
        Make sure the logit calculation in the detector is correct
        """
        torch.manual_seed(1234)
        n_dim = 20
        train = sample_dataset(centers=3, n_dim=n_dim, seed=123, n_samples=300)
        train_loader = DataLoader(train, batch_size=64, shuffle=True)

        model = ClassificationModel(num_inputs=n_dim, n_hidden=20)
        opti = SGD(model.parameters(), lr=0.01)

        criterion = CrossEntropyLoss()
        for epoch in range(2):
            for x, y in train_loader:
                opti.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                opti.step()

        x = torch.randn(size=(128, n_dim)) + torch.Tensor(n_dim * [0])

        model.eval()

        detector = ViM(
            model.features,
            d=5,
            w=model.classifier.weight.data,
            b=model.classifier.bias.data,
        )

        with torch.no_grad():
            logits1 = model(x)
            features = model.features(x)
        logits2 = detector._get_logits(features)
        for logit1, logit2 in zip(logits1.view(-1), logits2.view(-1)):
            self.assertAlmostEqual(logit1.item(), logit2.item(), places=5)


class ViMNumericsTest(unittest.TestCase):
    """Ill-conditioned features: the residual subspace must match a float64 eigendecomposition."""

    def test_ill_conditioned_features_match_float64_reference(self):
        torch.manual_seed(0)
        n, dim, c, d = 2000, 64, 4, 16
        scales = torch.logspace(0, -6, dim)
        rotation = torch.linalg.qr(torch.randn(dim, dim))[0]
        # ReLU features with a large mean, as penultimate CNN features: ViM centers at u, not at the mean
        mean = 5.0 * torch.rand(dim)
        features = ((torch.randn(n, dim) * scales) @ rotation.T + mean).relu()
        x = ((torch.randn(32, dim) * scales) @ rotation.T + 1.2 * mean).relu()
        w, b = torch.randn(c, dim), torch.randn(c)

        detector = ViM(None, d=d, w=w, b=b).fit_features(features, torch.zeros(n).long())
        residual = ((x - detector.u) @ detector.residual_subspace).norm(dim=1)

        u = -(torch.linalg.pinv(w.double()) @ b.double())
        centered = features.double() - u
        _, vecs = torch.linalg.eigh(centered.T @ centered / n)
        expected = ((x.double() - u) @ vecs[:, : dim - d]).norm(dim=1)

        torch.testing.assert_close(residual.double(), expected, rtol=1e-3, atol=1e-6)
