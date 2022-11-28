import unittest

import torch

from src.pytorch_ood.loss import MCHADLoss


class TestMCHAD(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        criterion = MCHADLoss(n_classes=10, n_dim=5)
        z = torch.randn(size=(128, 5))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        distances = criterion.distance(z)
        loss = criterion(distances, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

        dists = criterion.distance(z)
        self.assertEqual(dists.shape, (128, 10))
