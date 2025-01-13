import unittest

import torch

from src.pytorch_ood.detector import SHE
from src.pytorch_ood.model import WideResNet


class TestASH(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    @torch.no_grad()
    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = SHE(
            backbone=model.features,
            head=model.fc,
        )

        detector.fit_features(
            z=torch.randn(1000, 128),
            y=torch.arange(
                1000,
            )
            % 10,
            batch_size=128,
        )

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)
