import unittest

import torch

from src.pytorch_ood.model import VisionTransformer, WideResNet
from tests.helpers import for_examples


class TestPreTrainedModels(unittest.TestCase):
    """
    Download some pretrained models
    """

    @for_examples(("imagenet32", 1000), ("oe-cifar100-tune", 100), ("oe-cifar10-tune", 10))
    def test_wrn_pretrained(self, pretrain, n_classes):
        model = WideResNet(pretrained=pretrain, num_classes=n_classes)
        x = torch.ones(size=(1, 3, 32, 32))
        y = model(x)
        self.assertEqual(y.shape, (1, n_classes))

    def test_pretrained_vit(self):
        model = VisionTransformer(
            pretrained="b16-cifar10-tune", image_size=(384, 384), num_classes=10
        )
