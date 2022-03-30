import logging
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class ImageDatasetBase(VisionDataset):
    """
    Base Class for Downloading Image related Datasets

    Code inspired by : https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10
    """

    base_folder = None
    url = None
    filename = None
    tgz_md5 = None

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(ImageDatasetBase, self).__init__(
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
