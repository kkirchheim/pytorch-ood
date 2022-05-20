import unittest

import torch

from src.pytorch_ood.loss import EnergyRegularizedLoss


class TestEnergyRegularization(unittest.TestCase):
    """
    Test code for energy bounded learning
    """

    def test_forward(self):
        criterion = EnergyRegularizedLoss()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_positive(self):
        criterion = EnergyRegularizedLoss()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_negative(self):
        criterion = EnergyRegularizedLoss()
        logits = torch.randn(size=(128, 10))
        target = torch.ones(size=(128,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_set_alpha(self):
        criterion = EnergyRegularizedLoss(alpha=2)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
