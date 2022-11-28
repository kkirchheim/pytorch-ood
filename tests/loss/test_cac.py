import unittest

import torch

from src.pytorch_ood.loss import CACLoss


class TestCAC(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        criterion = CACLoss(n_classes=10)
        z = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(z, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

        dists = criterion.distance(z)
        self.assertEqual(dists.shape, (128, 10))

    def test_distance_calc(self):
        criterion = CACLoss(n_classes=10)
        z = torch.randn(size=(128, 10))
        distances = criterion.centers(z)
        self.assertEqual(distances.shape, (128, 10))

    def test_score(self):
        criterion = CACLoss(n_classes=10)
        z = torch.randn(size=(128, 10))
        distances = criterion.centers(z)
        s = criterion.score(distances)
        self.assertEqual(s.shape, (128,))
