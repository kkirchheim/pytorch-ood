"""
SVHN Dataset
"""
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10 as PTCIFAR10
from torchvision.datasets import CIFAR100 as PTCIFAR100
from torchvision.datasets import FashionMNIST
from torchvision.datasets import KMNIST as PTKMNIST
from torchvision.datasets import MNIST as PTMNIST
from torchvision.datasets import SVHN as PTSVHN

from oodtk.dataset.dataset import OSRVisionDataset


class SVHN(OSRVisionDataset):
    """
    """

    def __init__( self, root: str, download: bool = False, **kwargs) -> None:
        super(SVHN, self).__init__(root)
        dataset1 = PTSVHN(root, "train", download=download)
        dataset2 = PTSVHN(root, "test", download=download)
        self.dataset = ConcatDataset([dataset1, dataset2])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if type(img) is not Image.Image:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)

    @property
    def unique_targets(self) -> np.ndarray:
        return np.arange(0, stop=10)


class CIFAR10(OSRVisionDataset):
    """
    """

    def __init__(self, root: str, download: bool = False, **kwargs) -> None:
        super(CIFAR10, self).__init__(root)
        dataset1 = PTCIFAR10(root, train=True, download=download)
        dataset2 = PTCIFAR10(root, train=False, download=download)
        self.dataset = ConcatDataset([dataset1, dataset2])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if type(img) is not Image.Image:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)

    @property
    def unique_targets(self) -> np.ndarray:
        return np.arange(0, stop=10)


class CIFAR100(OSRVisionDataset):
    """
    """

    def __init__( self, root: str, download: bool = False, **kwargs) -> None:
        super(CIFAR100, self).__init__(root)
        dataset1 = PTCIFAR100(root, train=True, download=download)
        dataset2 = PTCIFAR100(root, train=False, download=download)
        self.dataset = ConcatDataset([dataset1, dataset2])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if type(img) is not Image.Image:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)

    @property
    def unique_targets(self) -> np.ndarray:
        return np.arange(0, stop=100)


class MNIST(OSRVisionDataset):
    """
    """

    def __init__( self, root: str, download: bool = False, **kwargs) -> None:
        super(MNIST, self).__init__(root)
        dataset1 = PTMNIST(root, train=True, download=download)
        dataset2 = PTMNIST(root, train=False, download=download)
        self.dataset = ConcatDataset([dataset1, dataset2])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if type(img) is not Image.Image:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)

    @property
    def unique_targets(self) -> np.ndarray:
        return np.arange(0, stop=10)


class KMNIST(OSRVisionDataset):
    """
    """

    def __init__( self, root: str, download: bool = False, **kwargs) -> None:
        super(KMNIST, self).__init__(root)
        dataset1 = PTKMNIST(root, train=True, download=download)
        dataset2 = PTKMNIST(root, train=False, download=download)
        self.dataset = ConcatDataset([dataset1, dataset2])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if type(img) is not Image.Image:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)

    @property
    def unique_targets(self) -> np.ndarray:
        return np.arange(0, stop=10)


class FMNIST(OSRVisionDataset):
    """
    """

    def __init__( self, root: str, download: bool = False, **kwargs) -> None:
        super(FMNIST, self).__init__(root)
        dataset1 = FashionMNIST(root, train=True, download=download)
        dataset2 = FashionMNIST(root, train=False, download=download)
        self.dataset = ConcatDataset([dataset1, dataset2])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index]

        if type(img) is not Image.Image:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataset)

    @property
    def unique_targets(self) -> np.ndarray:
        return np.arange(0, stop=10)
