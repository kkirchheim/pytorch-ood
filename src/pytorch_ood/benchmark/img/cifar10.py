"""

"""

from typing import List

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision.transforms import Compose

from pytorch_ood.benchmark import Benchmark
from pytorch_ood.dataset.img import (
    GaussianNoise,
    LSUNCrop,
    LSUNResize,
    Places365,
    Textures,
    TinyImageNet,
    TinyImageNetCrop,
    TinyImageNetResize,
    UniformNoise,
)
from pytorch_ood.utils import ToRGB, ToUnknown


class CIFAR10_ODIN(Benchmark):
    """
    Replicates the OOD detection benchmark from the ODIN paper for CIFAR 10.

    :see Paper: `ArXiv <https://arxiv.org/abs/1706.02690>`__

    Outlier datasets are

     * TinyImageNetCrop
     * TinyImageNetResize
     * LSUNResize
     * LSUNCrop
     * Uniform
     * Gaussian
    """

    def __init__(self, root, transform):
        """
        :param root: where to store datasets
        :param transform: transform to apply to images
        """
        self.transform = transform
        self.train_in = CIFAR10(root, download=True, transform=transform, train=True)
        self.test_in = CIFAR10(root, download=True, transform=transform, train=False)

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
                seed=123,
            ),
            GaussianNoise(
                1000,
                size=(32, 32, 3),
                transform=transform,
                target_transform=ToUnknown(),
                seed=123,
            ),
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    def train_set(self) -> Dataset:
        """
        Training dataset
        """
        return self.train_in

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.
        If known and unknown are true, each dataset contains ID and OOD data.

        :param known: include ID
        :param unknown: include OOD
        """

        if known and unknown:
            return [self.test_in + other for other in self.test_oods]

        if known and not unknown:
            return [self.train_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()


class CIFAR10_OpenOOD(Benchmark):
    """
    Replicates the CIFAR-10 benchmark proposed in
    *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection*.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__

    Near-OOD datasets:

     * CIFAR-100
     * TinyImageNet

    Far-OOD datasets:

     * MNIST
     * SVHN
     * Textures
     * Places365

    """

    def __init__(self, root, transform):
        """
        :param root: where to store datasets
        :param transform: transform to apply to images
        """
        self.transform = Compose([ToRGB(), transform])
        self.train_in = CIFAR10(root, download=True, transform=transform, train=True)
        self.test_in = CIFAR10(root, download=True, transform=transform, train=False)

        self.test_oods = [
            CIFAR100(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
                train=False,
            ),
            TinyImageNet(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
                subset="val",
            ),
            MNIST(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
                train=False,
            ),
            SVHN(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
                split="test",
            ),
            Textures(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            Places365(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    def train_set(self) -> Dataset:
        """
        Training dataset
        """
        return self.train_in

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.
        If known and unknown are true, each dataset contains ID and OOD data.

        :param known: include ID
        :param unknown: include OOD
        """

        if known and unknown:
            return [self.test_in + other for other in self.test_oods]

        if known and not unknown:
            return [self.train_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()
