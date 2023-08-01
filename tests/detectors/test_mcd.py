import unittest

import torch

from src.pytorch_ood.detector import MCD
from tests.helpers import ClassificationModel, SegmentationModel


class TestMCD(unittest.TestCase):
    """
    Tests for Monte Carlo Dropout
    """

    def test_something(self):
        model = ClassificationModel()
        mcd = MCD(model, mode="var")

        x = torch.randn(size=(128, 10))
        y = mcd.predict(x)

        print(y)
        self.assertEqual(torch.Size([128]), y.shape)
        self.assertIsNotNone(y)

    def test_segmentation(self):
        model = SegmentationModel()
        mcd = MCD(model)

        x = torch.zeros(size=(4, 3, 32, 32))
        y = mcd.predict(x)

        print(y.shape)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (4, 32, 32))

    def test_mean_eq_mean(self):
        model = ClassificationModel()
        mcd = MCD(model, mode="mean", samples=1000)
        x = torch.randn(size=(128, 10))

        y = mcd.predict(x)

        mean = -mcd.run(model, x, samples=1000)[0]
        self.assertTrue(torch.isclose(y, mean, rtol=0.05).all())
