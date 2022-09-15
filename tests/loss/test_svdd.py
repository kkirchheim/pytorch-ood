import unittest

import torch

from src.pytorch_ood.loss import DeepSVDDLoss, SSDeepSVDDLoss


class TestDeepSVDD(unittest.TestCase):
    def test_forward(self):
        criterion = DeepSVDDLoss(n_features=10, reduction=None)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertTrue((loss[5:] == 0).all())


class TestSSDeepSVDD(unittest.TestCase):
    def test_forward(self):
        criterion = SSDeepSVDDLoss(n_features=10)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)

    def test_forward_2(self):
        criterion = SSDeepSVDDLoss(n_features=10, reduction=None)
        logits = torch.randn(size=(10, 10))
        target = -1 * torch.ones(size=(10,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertFalse((loss == 0).any())
