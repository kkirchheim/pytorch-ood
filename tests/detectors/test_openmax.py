import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import OpenMax
from tests.helpers import ClassificationModel


class TestOpenMax(unittest.TestCase):
    """
    Tests for Monte Carlo Dropout
    """

    def test_something(self):
        model = ClassificationModel(num_outputs=3)
        openmax = OpenMax(model)

        x = torch.randn(size=(128, 10))
        y = torch.arange(128) % 3

        loader = DataLoader(TensorDataset(x, y))

        openmax.fit(loader)

        scores = openmax.predict(x)

        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape, (128,))

    def test_predict_before_fit(self):
        openmax = OpenMax(ClassificationModel(num_outputs=3))
        x = torch.randn(size=(4, 10))

        with self.assertRaises(RequiresFittingException):
            openmax.predict(x)

        with self.assertRaises(RequiresFittingException):
            openmax.predict_logits(torch.randn(size=(4, 3)))
