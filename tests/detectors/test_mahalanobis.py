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
