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
