import unittest

import torch

from src.pytorch_ood.detector import ASH
from src.pytorch_ood.model import WideResNet
from tests.helpers import ClassificationModel, sample_dataset


class TestASH(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = ASH(
            backbone=model.features_before_pool,
            head=model.forward_from_before_pool,
        )

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)
