import unittest

import torch

from src.pytorch_ood.loss import EntropicOpenSetLoss
from tests.helpers.model import SegmentationModel


class TestEntropicLoss(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        target = torch.arange(0, 128).long() % 10
        target[50:] = -1

        criterion = EntropicOpenSetLoss()
        x = torch.randn(size=(128, 10))
        loss = criterion(x, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_all_ood(self):
        target = torch.ones(0, 128).long() * -1

        criterion = EntropicOpenSetLoss()
        x = torch.randn(size=(128, 10))
        loss = criterion(x, target)

        self.assertIsNotNone(loss)
        self.assertEqual(loss, 0)

    def test_segmentation_with_unknown(self):
        model = SegmentationModel()
        criterion = EntropicOpenSetLoss(reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertNotEqual(loss[0, 0, 0], 0)
        loss.mean().backward()
