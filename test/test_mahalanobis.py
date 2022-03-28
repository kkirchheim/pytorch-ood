import unittest

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from oodtk import Mahalanobis
from oodtk.dataset.img import UniformNoise


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MyTestCase(unittest.TestCase):
    @unittest.skip("Not fully implemented")
    def test_something(self):
        nn = Model()
        model = Mahalanobis(nn)

        data = DataLoader(MNIST(root="/tmp/", download=True, transform=ToTensor()))

        model.fit(data)
        self.assertEqual(True, True)
