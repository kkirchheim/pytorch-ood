import unittest

import torch

from src.pytorch_ood.loss import CrossEntropyLoss
from tests.helpers import ClassificationModel, SegmentationModel


class TestCrossEntropyLoss(unittest.TestCase):
    """
    Test code for the cross-entropy loss
    """

    def test_example_1(self):
        """
        Mix of IN and OOD samples
        """
        criterion = CrossEntropyLoss()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_example_2(self):
        criterion = CrossEntropyLoss()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_return_0_on_all_unknown(self):
        criterion = CrossEntropyLoss()
        logits = torch.randn(size=(10, 10))
        target = torch.ones(size=(10,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertEqual(loss, 0)

    def test_backwards(self):
        model = ClassificationModel()
        criterion = CrossEntropyLoss()
        x = torch.randn(size=(10, 10))
        logits = model(x)
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        loss.backward()

    # @unittest.skip("Not implemented")
    def test_segmentation(self):
        model = SegmentationModel()
        criterion = CrossEntropyLoss()
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()

    def test_segmentation_with_unknown(self):
        model = SegmentationModel()
        criterion = CrossEntropyLoss(reduction=None)
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)

        self.assertEqual(loss[0, 0, 0], 0)
        self.assertNotEqual(loss[0, 0, 1], 0)
        loss.mean().backward()
