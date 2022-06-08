import unittest

import torch

from src.pytorch_ood.loss import CenterLoss
from tests.helpers import ClassificationModel


class TestCenterLoss(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        target = torch.arange(0, 128).long() % 10
        target[50:] = -1

        criterion = CenterLoss(n_classes=10, n_dim=8)
        z = torch.randn(size=(128, 8))
        d = criterion.centers(z)
        loss = criterion(d, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_distance_calc(self):
        criterion = CenterLoss(n_classes=10, n_dim=8)
        z = torch.randn(size=(128, 8))
        distances = criterion.centers(z)
        self.assertEqual(distances.shape, (128, 10))

    def test_backward(self):
        criterion = CenterLoss(n_classes=10, n_dim=8)
        model = ClassificationModel(num_outputs=8)
        target = torch.arange(0, 128).long() % 10
        target[50:] = -1
        x = torch.randn(size=(128, 10))

        z = model(x)

        d = criterion.centers(z)
        loss = criterion(d, target)
        loss.backward()
        self.assertGreaterEqual(loss, 0)

    def test_all_ood(self):
        target = torch.ones(0, 128).long() * -1

        criterion = CenterLoss(n_classes=10, n_dim=8)
        z = torch.randn(size=(128, 8))
        d = criterion.centers(z)
        loss = criterion(d, target)

        self.assertIsNotNone(loss)
        self.assertEqual(loss, 0)
