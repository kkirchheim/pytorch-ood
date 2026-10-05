""" """

from typing import List

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from pytorch_ood.api import BenchmarkInfo, Paper, Task
from pytorch_ood.benchmark import Benchmark
from pytorch_ood.dataset.img import (
    GaussianNoise,
    LSUNCrop,
    LSUNResize,
    TinyImageNetCrop,
    TinyImageNetResize,
    UniformNoise,
)
from pytorch_ood.utils import ToUnknown


class CIFAR100_ODIN(Benchmark):
    """
    Replicates the OOD detection benchmark from the ODIN paper for CIFAR 100.

    Outlier datasets are

    * TinyImageNetCrop
    * TinyImageNetResize
    * LSUNResize
    * LSUNCrop
    * Uniform noise
    * Gaussian noise

    The entries of ``ood_names`` are ``TinyImageNetCrop``, ``TinyImageNetResize``, ``LSUNResize``, ``LSUNCrop``, ``Uniform``, ``Gaussian``.
    """

    info = BenchmarkInfo(
        paper=Paper(
            title="Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks",
            venue="ICLR",
            year=2018,
            url="https://arxiv.org/abs/1706.02690",
            code="https://github.com/facebookresearch/odin",
        ),
        tasks={Task.CLASSIFICATION},
    )

    def __init__(self, root, transform):
        """
        :param root: where to store datasets
        :param transform: transform to apply to images
        """
        self.transform = transform
        self.train_in = CIFAR100(root, download=True, transform=transform, train=True)
        self.test_in = CIFAR100(root, download=True, transform=transform, train=False)

        self.test_oods = [
            TinyImageNetCrop(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            TinyImageNetResize(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            LSUNResize(root, download=True, transform=transform, target_transform=ToUnknown()),
            LSUNCrop(root, download=True, transform=transform, target_transform=ToUnknown()),
            UniformNoise(
                1000,
                size=(32, 32, 3),
                transform=transform,
                target_transform=ToUnknown(),
            ),
            GaussianNoise(
                1000,
                size=(32, 32, 3),
                transform=transform,
                target_transform=ToUnknown(),
            ),
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [
            "TinyImageNetCrop",
            "TinyImageNetResize",
            "LSUNResize",
            "LSUNCrop",
            "Uniform",
            "Gaussian",
        ]

    def train_set(self) -> Dataset:
        """
        Training dataset
        """
        return self.train_in

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.

        :param known: include ID
        :param unknown: include OOD
        :return: with ``known`` and ``unknown``, one dataset per entry of ``ood_names`` that
            combines the ID test set with that OOD set. With ``unknown`` only, the OOD sets in the
            order of ``ood_names``. With ``known`` only, the ID test set, once.
        :raises ValueError: if both ``known`` and ``unknown`` are false
        """

        if known and unknown:
            return [self.test_in + other for other in self.test_oods]

        if known and not unknown:
            return [self.test_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError("At least one of `known` or `unknown` must be True")
