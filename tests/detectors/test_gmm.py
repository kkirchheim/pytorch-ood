import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException, ModelNotSetException
from src.pytorch_ood.detector import GMM
from tests.helpers import ClassificationModel


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
        detector = GMM(model=None)

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
        detector = GMM(model=None)

        z = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        detector.fit_features(z, y)

        with self.assertRaises(ModelNotSetException):
            detector(torch.randn(size=(5, 10)))
