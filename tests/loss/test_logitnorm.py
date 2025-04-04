import unittest

import torch

from src.pytorch_ood.loss import LogitNorm

torch.manual_seed(123)


class TestLogitNorm(unittest.TestCase):
    """
    Test code for energy bounded learning
    """

    def test_forward(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)

    def test_forward_only_positive(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)

    def test_forward_only_negative(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(128, 10))
        target = torch.ones(size=(128,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)

    def test_set_alpha(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)

    def test_set_ms(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
