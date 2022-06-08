import unittest

import torch

from src.pytorch_ood.loss import IILoss
from tests.helpers import ClassificationModel


class TestIILoss(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        target = torch.arange(0, 128).long() % 10
        target[50:] = -1

        criterion = IILoss(n_classes=10, n_embedding=8)
        z = torch.randn(size=(128, 8))
        loss = criterion(z, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
