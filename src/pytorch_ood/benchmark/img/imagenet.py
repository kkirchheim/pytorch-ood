from typing import List

from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose

from pytorch_ood.benchmark import Benchmark
from pytorch_ood.dataset.img import (
    NINCO,
    ImageNet200,
    OpenImagesO,
    SSBHard,
    Textures,
    iNaturalist,
)
from pytorch_ood.utils import ToRGB, ToUnknown


class ImageNet_OpenOOD(Benchmark):
    """
    Replicates the ImageNet benchmark proposed in
    *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection*.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__

    Near-OOD datasets:

     * SSB-Hard
     * NINCO

    Far-OOD datasets:

     * iNaturalist
     * Textures
     * OpenImage-O

    """

    def __init__(self, root, image_net_root, transform):
        """
        :param root: where to store datasets
        :param image_net_root: root for the ImageNet dataset
        :param transform: transform to apply to images
        """
        self.transform = Compose([ToRGB(), transform])
        self._train_in = None
        self.image_net_root = image_net_root
        self.test_in = ImageNet(image_net_root, transform=self.transform, split="val")

        self.test_oods = [
            SSBHard(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            NINCO(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            iNaturalist(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            Textures(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            OpenImagesO(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    @property
    def train_in(self):
        # lazy loading only if needed
        if not self._train_in:
            self._train_in = ImageNet(self.image_net_root, split="train", transform=self.transform)

        return self._train_in

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


class ImageNet200_OpenOOD(Benchmark):
    """
    Replicates the ImageNet-200 benchmark proposed in
    *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection*.

    .. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
       :alt: slop-badge

    The in-distribution data is the 200-class :class:`ImageNet200
    <pytorch_ood.dataset.img.ImageNet200>` subset of ImageNet-1K (classes identical to
    ImageNet-R). This benchmark therefore requires a classifier trained on these 200 classes;
    a standard 1000-class ImageNet model will not work. The out-of-distribution datasets are
    identical to those of :class:`ImageNet_OpenOOD`.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__

    Near-OOD datasets:

     * SSB-Hard
     * NINCO

    Far-OOD datasets:

     * iNaturalist
     * Textures
     * OpenImage-O

    """

    def __init__(self, root, image_net_root, transform):
        """
        :param root: where to store the OOD datasets
        :param image_net_root: root of a torchvision-compatible ImageNet directory
        :param transform: transform to apply to images
        """
        self.transform = Compose([ToRGB(), transform])
        self._train_in = None
        self.image_net_root = image_net_root
        self.test_in = ImageNet200(image_net_root, split="test", transform=self.transform)
        self.val_in = ImageNet200(image_net_root, split="val", transform=self.transform)

        self.test_oods = [
            SSBHard(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            NINCO(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            iNaturalist(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            Textures(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
            OpenImagesO(
                root,
                download=True,
                transform=self.transform,
                target_transform=ToUnknown(),
            ),
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    @property
    def train_in(self):
        # lazy loading only if needed
        if not self._train_in:
            self._train_in = ImageNet200(
                self.image_net_root, split="train", transform=self.transform
            )

        return self._train_in

    def train_set(self) -> Dataset:
        """
        Training dataset (in-distribution)
        """
        return self.train_in

    def validation_set(self) -> Dataset:
        """
        In-distribution validation dataset (1000 held-out images, 5 per class), as used
        by OpenOOD for hyperparameter tuning.
        """
        return self.val_in

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
            return [self.test_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()
