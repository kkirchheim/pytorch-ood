import unittest

import torch
from torch.utils.data import DataLoader

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import TemperatureScaling
from tests.helpers import ClassificationModel, sample_dataset


class TestTScaling(unittest.TestCase):
    """
    Tests for Temperature Scaling
    """

    def test_requires_fitting(self):
        model = ClassificationModel()
        detector = TemperatureScaling(model)

        x = torch.zeros(size=(128, 10))

        with self.assertRaises(RequiresFittingException):
            y = detector(x)

    def test_input(self):
        """ """
        model = ClassificationModel()
        detector = TemperatureScaling(model)

        x = torch.randn(size=(128, 10))
        y = torch.randint(0, 2, size=(x.shape[0],))
        detector.fit_features(x, y)

        print(detector.t)
        self.assertIsNotNone(detector.t)

        scores = detector.predict_features(x)
        self.assertIsNotNone(scores)

    def test_input2(self):
        """ """
        model = ClassificationModel()
        detector = TemperatureScaling(model)

        ds = sample_dataset(n_dim=10)
        loader = DataLoader(ds)

        x = torch.randn(size=(128, 10))

        scores = detector.fit(loader).predict(x)

        print(detector.t)
        self.assertIsNotNone(detector.t)
        self.assertIsNotNone(scores)
