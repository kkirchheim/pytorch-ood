import unittest

import torch
from torch.optim import SGD

from src.pytorch_ood.loss import DeepSVDDLoss, SSDeepSVDDLoss
from tests.helpers import ClassificationModel


class TestDeepSVDD(unittest.TestCase):
    def test_forward(self):
        criterion = DeepSVDDLoss(n_dim=10, reduction=None)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertTrue((loss[5:] == 0).all())

    def test_radius(self):
        model = ClassificationModel(n_hidden=1024)
        opti = SGD(model.parameters(), lr=0.01)
        criterion = DeepSVDDLoss(n_dim=3, radius=1.0, reduction=None)

        x = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()

        target[5:] = -1

        for i in range(1000):
            loss = criterion(model(x), target)

            loss.mean().backward()
            opti.zero_grad()
            opti.step()

        print(loss)
        print(criterion.distance(model(x)))

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
