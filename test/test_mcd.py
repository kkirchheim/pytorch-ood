import unittest

import torch

from pytorch_ood import MCD


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.p = torch.nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)


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
        self.assertEqual(True, True)
