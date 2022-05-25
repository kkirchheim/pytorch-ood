import unittest

import torch

from src.pytorch_ood.loss import DeepSVDDLoss


class TestDeepSVDD(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        criterion = DeepSVDDLoss(n_features=10)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
