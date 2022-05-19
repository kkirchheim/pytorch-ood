import unittest
from test.helpers import ClassificationModel, SegmentationModel

import torch

from pytorch_ood.detector import NegativeEnergy


class TestEnergy(unittest.TestCase):
    """
    Tests for Energy based Out-of-Distribution Detection
    """

    def test_classification_input(self):
        model = ClassificationModel()
        energy = NegativeEnergy(model)

        x = torch.zeros(size=(128, 10))
        y = energy.predict(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))

    def test_segmentation_input(self):
        """
        Tests input map for semantic segmentation
        """
        model = SegmentationModel()
        energy = NegativeEnergy(model)

        x = torch.zeros(size=(128, 3, 32, 32))
        y = energy.predict(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128, 32, 32))
