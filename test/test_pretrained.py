import unittest
from test.helpers import for_examples

from oodtk.model import VisionTransformer, WideResNet


class TestPreTrainedModels(unittest.TestCase):
    """
    Download some pretrained models
    """

    @for_examples(("imagenet32", 1000), ("oe-cifar100-tune", 100), ("oe-cifar10-tune", 10))
    def test_wrn_pretrained(self, pretrain, n_classes):
        model = WideResNet.from_pretrained(pretrain, num_classes=n_classes)

    def test_pretrained_vit(self):
        model = VisionTransformer.from_pretrained(
            "b16-cifar10-tune", image_size=(384, 384), num_classes=10
        )
