import unittest
from test.helpers import for_examples

from oodtk.model import WideResNetPretrained


class TestPreTrainedModels(unittest.TestCase):
    """
    Download some pretrained models
    """

    @for_examples(("imagenet32", 1000), ("oe-cifar100-tune", 100), ("oe-cifar10-tune", 10))
    def test_wrn_pretrained(self, pretrain, n_classes):
        model = WideResNetPretrained(pretrain=pretrain, num_classes=n_classes)
