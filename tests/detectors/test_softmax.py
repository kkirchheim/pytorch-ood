import unittest

import torch

from src.pytorch_ood.detector import MaxSoftmax
from tests.helpers import ClassificationModel, SegmentationModel


class TestSoftmax(unittest.TestCase):
    """
    Tests for Energy based Out-of-Distribution Detection
    """

    def test_classification_input(self):
        model = ClassificationModel()
        detector = MaxSoftmax(model)

        x = torch.zeros(size=(128, 10))
        with torch.no_grad():
            y = detector(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))

    def test_segmentation_input(self):
        """
        Tests input map for semantic segmentation
        """
        model = SegmentationModel()
        detector = MaxSoftmax(model)

        x = torch.zeros(size=(128, 3, 32, 32))

        with torch.no_grad():
            y = detector(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128, 32, 32))
