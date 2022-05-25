import unittest

import torch

from src.pytorch_ood.loss import ObjectosphereLoss


class TestObjectosphere(unittest.TestCase):
    """
    Test code of examples
    """

    def test_example_1(self):
        criterion = ObjectosphereLoss(alpha=1.0, xi=1.0)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0)
