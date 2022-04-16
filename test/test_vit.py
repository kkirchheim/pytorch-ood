import unittest

import torch

from pytorch_ood.model.vit import VisionTransformer


class VisionTransformerTest(unittest.TestCase):
    """
    Test the vision transformer
    """

    def test_forward(self):
        model = VisionTransformer(num_layers=2)
        x = torch.randn((2, 3, 256, 256))
        out = model(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1000)

    def test_forward_2(self):
        model = VisionTransformer(num_layers=2, num_classes=10, image_size=(32, 32))
        x = torch.randn((2, 3, 32, 32))
        out = model(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 10)
