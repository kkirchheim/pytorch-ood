import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from oodtk import Mahalanobis


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.p = torch.nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p(x)


class MahalanobisTest(unittest.TestCase):
    """ """

    def test_something(self):
        nn = Model()
        model = Mahalanobis(nn)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(scores)

        scores = model(torch.ones(size=(10, 10)) * 10)
        print(scores)
        self.assertEqual(True, True)
