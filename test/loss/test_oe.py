import unittest

import torch

from pytorch_ood.loss import OutlierExposureLoss


class TestOutlierExposure(unittest.TestCase):
    """
    Test code of examples
    """

    def test_example_1(self):
        criterion = OutlierExposureLoss()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_example_2(self):
        criterion = OutlierExposureLoss()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_example_3(self):
        criterion = OutlierExposureLoss()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long() * -1
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
