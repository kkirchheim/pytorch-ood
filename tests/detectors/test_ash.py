import unittest

import torch

from src.pytorch_ood.detector import ASH
from src.pytorch_ood.model import WideResNet


class TestASH(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = ASH(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)
