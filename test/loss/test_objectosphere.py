import unittest

import torch

from oodtk.loss import ObjectosphereLoss


class TestObjectosphere(unittest.TestCase):
    """
    Test code of examples
    """

    def test_example_1(self):
        criterion = ObjectosphereLoss(lambda_=1.0, zetta=1.0)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
