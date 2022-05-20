import unittest

import torch
from torch.utils.data import TensorDataset

from src.pytorch_ood.dataset.ossim import DynamicOSS


class TestOSSIM(unittest.TestCase):
    """
    Test code of examples
    """

    def test_something(self):
        x = torch.randn(size=(100, 10))
        y = torch.arange(100) % 10

        dataset = TensorDataset(x, y)

        ossim = DynamicOSS(dataset, kuc=0, uuc_test=4, uuc_val=0, seed=0)

        self.assertEqual(6, len(ossim.kkc))
        self.assertEqual(4, len(ossim.uuc))

        self.assertIsNotNone(ossim.test_dataset())
        self.assertIsNotNone(ossim.train_dataset())
        self.assertIsNotNone(ossim.val_dataset())
