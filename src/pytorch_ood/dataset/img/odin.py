"""
Datasets used for testing in ODIN

:see  https://github.com/facebookresearch/odin
"""
import logging
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class TinyImageNetCrop(VisionDataset):
    """
    Cropped version of the tiny imagenet
    """

    base_folder = "Imagenet/test/"
    url = "https://www.dropbox.com/s/raw/avgm2u562itwpkl/Imagenet.tar.gz"
    filename = "Imagenet.tar.gz"
    tgz_md5 = "7c0827e4246c3718a5ee076e999e52e5"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(TinyImageNetCrop, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = os.listdir(self.basedir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file, target = self.files[index], -1
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        path = os.path.join(self.root, self.base_folder, file)
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.files)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, self.tgz_md5)

    def download(self) -> None:
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    @property
    def train(self):
        return False


class TinyImageNetResize(TinyImageNetCrop):
    """
    Resized version of the tiny imagenet

    """

    base_folder = "Imagenet_resize/Imagenet_resize/"
    url = "https://www.dropbox.com/s/raw/kp3my3412u5k9rl/Imagenet_resize.tar.gz"
    filename = "Imagenet_resize.tar.gz"
    tgz_md5 = "0f9ff11d45babf2eff5fe12281d1ac31"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(TinyImageNetResize, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @property
    def train(self):
        return False


class LSUNCrop(TinyImageNetCrop):
    """
    Cropped version of the LSUN

    """

    base_folder = "LSUN/test/"
    url = "https://www.dropbox.com/s/raw/fhtsw1m3qxlwj6h/LSUN.tar.gz"
    filename = "LSUN.tar.gz"
    tgz_md5 = "458a0a0ab8e5f1cb4516d7400568e460"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(LSUNCrop, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @property
    def train(self):
        return False


class LSUNResize(TinyImageNetCrop):
    """
    Resized version of the LSUN dataset

    """

    base_folder = "LSUN_resize/LSUN_resize"
    url = "https://www.dropbox.com/s/raw/moqh2wh8696c3yl/LSUN_resize.tar.gz"
    filename = "LSUN_resize.tar.gz"
    tgz_md5 = "278b7b31c8cb7e804a1465a8ce03a2dc"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(LSUNResize, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @property
    def train(self):
        return False
