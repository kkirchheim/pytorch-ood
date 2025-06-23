import unittest

import torch

from src.pytorch_ood.detector import GradNorm
from src.pytorch_ood.model import WideResNet


class TestGradNorm(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = GradNorm(model, param_filter=lambda x: x.startswith("fc."))

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (16,))
