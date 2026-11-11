import unittest

import torch

from pytorch_ood.loss import BackgroundClassLoss


class TestBackgroundClass(unittest.TestCase):
    """
    Test Background Class Loss
    """

    def test_class_to_high(self):
        criterion = BackgroundClassLoss(n_classes=3)
        logits = torch.randn(size=(10, 3))
        target = torch.arange(10) % 3
        target[5:] = 10

        with self.assertRaises(ValueError):
            criterion(logits, target)
