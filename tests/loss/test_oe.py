import unittest

import torch

from src.pytorch_ood.loss import OutlierExposureLoss
from tests.helpers.model import SegmentationModel


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
        target = torch.ones(size=(10,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_segmentation_with_unknown(self):
        model = SegmentationModel()
        criterion = OutlierExposureLoss(reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        print(loss)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertNotEqual(loss[0, 0, 0], 0)
        loss.mean().backward()
