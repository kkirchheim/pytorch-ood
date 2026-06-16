import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import MultiMahalanobis
from tests.helpers import ClassificationModel
from tests.helpers.model import ConvClassifier


class MultiMahalanobisTest(unittest.TestCase):
    """
    Test multi-layer mahalanobis method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_something(self):
        nn = ConvClassifier(in_channels=3, out_channels=16)
        model = MultiMahalanobis([nn.layer1, nn.pool])

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 3, 16, 16))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(f"Scores: {scores}")

        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape[0], 20)

    def test_nofit(self):
        nn = ClassificationModel()
        model = MultiMahalanobis([nn.layer1, nn.dropout])
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            model(x)

    def test_refit_does_not_accumulate(self):
        """Re-fitting replaces parameters instead of appending (e.g. during HPO)."""
        nn = ConvClassifier(in_channels=3, out_channels=16)
        layers = [nn.layer1, nn.pool]
        model = MultiMahalanobis(layers)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 3, 16, 16))
        loader = DataLoader(TensorDataset(x, y))

        model.fit(loader)
        first_mu = [m.clone() for m in model.mu]

        model.fit(loader)

        # one set of parameters per layer, not accumulated across fits
        self.assertEqual(len(model.mu), len(layers))
        self.assertEqual(len(model.cov), len(layers))
        self.assertEqual(len(model.precision), len(layers))
        # same data => same fitted centers
        for a, b in zip(first_mu, model.mu):
            self.assertTrue(torch.allclose(a, b))

        scores = model(x)
        self.assertEqual(scores.shape[0], 20)
