import unittest

import torch

from src.pytorch_ood.loss import BackgroundClassLoss
from tests.helpers import ClassificationModel


class TestBackgroundClass(unittest.TestCase):
    """
    Test Background Class Loss
    """

    def test_example_1(self):
        criterion = BackgroundClassLoss(n_classes=3)
        model = ClassificationModel(num_outputs=4)
        input = torch.randn(size=(10, 10))
        target = torch.arange(10) % 3
        target[5:] = -1

        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_class_to_high(self):
        criterion = BackgroundClassLoss(n_classes=3)
        logits = torch.randn(size=(10, 3))
        target = torch.arange(10) % 3
        target[5:] = 10

        with self.assertRaises(ValueError):
            criterion(logits, target)
