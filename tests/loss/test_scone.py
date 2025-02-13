import unittest

import torch

from src.pytorch_ood.loss import EnergyMarginLoss

torch.manual_seed(123)

logistic_regression = torch.nn.Linear(1, 1)


class TestEnergyMargin(unittest.TestCase):
    """
    Test code for margin-energy-based optimization
    """

    def test_forward(self):
        criterion = EnergyMarginLoss(full_train_loss=0)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        # At least one sample has to be OOD; otherwise, energy_loss_out returns NaN
        target[0:1] = -1

        loss = criterion(logits, target, logistic_regression)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_backward(self):
        criterion = EnergyMarginLoss(full_train_loss=0)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target, logistic_regression)
        loss.backward()

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_inccreasing_eta(self):
        criterion = EnergyMarginLoss(full_train_loss=0, eta=1.0)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        low_eta_loss = criterion(logits, target, logistic_regression)

        self.assertIsNotNone(low_eta_loss)
        self.assertGreater(low_eta_loss, 0)

        criterion = EnergyMarginLoss(full_train_loss=0, eta=5.0)

        high_eta_loss = criterion(logits, target, logistic_regression)

        self.assertIsNotNone(high_eta_loss)
        self.assertGreater(high_eta_loss, 0)

        # introducing wider margin should minimize the loss further
        self.assertGreater(low_eta_loss, high_eta_loss)
