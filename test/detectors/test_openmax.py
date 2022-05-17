import unittest
from test.helpers import ClassificationModel

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detectors.openmax import OpenMax


class TestOpenMax(unittest.TestCase):
    """
    Tests for Monte Carlo Dropout
    """

    def test_something(self):
        model = ClassificationModel(num_classes=3)
        openmax = OpenMax(model)

        x = torch.randn(size=(128, 10))
        y = torch.arange(128) % 3

        loader = DataLoader(TensorDataset(x, y))

        openmax.fit(loader)

        scores = openmax.predict(x)

        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape, (128,))
