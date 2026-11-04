import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import Mahalanobis, MultiMahalanobis
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


class MultiMahalanobisScoreTest(unittest.TestCase):
    def test_weighted_sum_of_half_squared_distances_to_closest_center(self):
        detector = MultiMahalanobis([torch.nn.Identity(), torch.nn.Identity()], alpha=[1.0, 2.0])
        detector.mu = [
            torch.tensor([[0.0, 0.0], [4.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
        ]
        detector.precision = [torch.diag(torch.tensor([1.0, 4.0])), torch.eye(2)]
        x = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
        # layer 1: squared distances (5, 13) and (9, 1); layer 2: (2, 2) and (9, 13)
        expected = 1.0 * torch.tensor([2.5, 0.5]) + 2.0 * torch.tensor([1.0, 4.5])
        torch.testing.assert_close(detector.predict_structured([x, x]), expected)


class MultiMahalanobisSingleLayerTest(unittest.TestCase):
    def test_single_layer_equals_mahalanobis(self):
        """One identity layer with weight 1 fits the same Gaussian as Mahalanobis."""
        torch.manual_seed(0)
        y = torch.arange(60) % 3
        z = torch.randn(60, 80) + torch.randn(3, 80)[y]  # fewer samples than dimensions
        x = torch.randn(16, 80)
        for kwargs in [{}, {"eps": 1e-2}, {"use_pinv": True}]:
            with self.subTest(**kwargs):
                multi = MultiMahalanobis([torch.nn.Identity()], **kwargs).fit_structured([z], y)
                single = Mahalanobis(None, **kwargs).fit_features(z, y)
                torch.testing.assert_close(multi.precision[0], single.precision)
                torch.testing.assert_close(
                    multi.predict_structured([x]), single.predict_features(x)
                )
