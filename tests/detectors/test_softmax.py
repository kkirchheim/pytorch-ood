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

    def test_temperature_is_float(self):
        detector = MaxSoftmax(None, t=2)

        self.assertIsInstance(detector.t, float)
        self.assertEqual(detector.t, 2.0)

    def test_temperature_scales_logits(self):
        logits = torch.randn(16, 5)
        detector = MaxSoftmax(None, t=2.0)

        expected = -(logits / 2.0).softmax(dim=1).max(dim=1).values
        torch.testing.assert_close(detector.predict_logits(logits), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_model_without_to(self):
        # the temperature must not pin the detector to the CPU when the model is on the GPU
        model = ClassificationModel().cuda()
        detector = MaxSoftmax(model)

        x = torch.zeros(size=(8, 10), device="cuda")
        with torch.no_grad():
            y = detector(x)
        self.assertEqual(y.device, x.device)
        self.assertEqual(y.shape, (8,))
