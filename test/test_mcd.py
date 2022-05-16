import unittest
from test.helpers import Model

import torch

from pytorch_ood import MCD


class TestMCD(unittest.TestCase):
    """
    Tests for Monte Carlo Dropout
    """

    def test_something(self):
        model = Model()
        mcd = MCD(model)

        x = torch.zeros(size=(1, 10))
        y = mcd.predict(x)
        self.assertIsNotNone(y)
