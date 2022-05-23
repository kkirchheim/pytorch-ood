import unittest

import torch

from src.pytorch_ood.loss import ConfidenceLoss


class TestConfidenceLoss(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        target = torch.arange(0, 128).long() % 10
        target[50:] = -1

        criterion = ConfidenceLoss()
        x = torch.randn(size=(128, 10))
        c = torch.randn(size=(128, 1))
        loss = criterion(x, c, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_all_ood(self):
        target = torch.ones(0, 128).long() * -1

        print(target)
        criterion = ConfidenceLoss()
        x = torch.randn(size=(128, 10))
        c = torch.randn(size=(128, 1))
        loss = criterion(x, c, target)

        self.assertIsNotNone(loss)
        self.assertEqual(loss, 0)
