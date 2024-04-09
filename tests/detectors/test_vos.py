import unittest

import torch

from src.pytorch_ood.detector import VOSBased
from tests.helpers import ClassificationModel, SegmentationModel

torch.manual_seed(123)


class TestEnergy(unittest.TestCase):
    """
    Tests for Energy based Out-of-Distribution Detection
    """

    def init_detektor(self, model, num_classes):
        weights_energy = torch.nn.Linear(num_classes, 1).cpu()
        torch.nn.init.uniform_(weights_energy.weight)
        detector = VOSBased(model, weights_energy)
        return detector

    def test_classification_input(self):
        model = ClassificationModel()
        detector = self.init_detektor(model, num_classes=3)

        x = torch.zeros(size=(128, 10))
        y = detector(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))

    def test_segmentation_input(self):
        """
        Tests input map for semantic segmentation
        """
        model = SegmentationModel()
        detector = self.init_detektor(model, num_classes=3)

        x = torch.zeros(size=(128, 3, 32, 32))
        y = detector(x)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128, 32, 32))
