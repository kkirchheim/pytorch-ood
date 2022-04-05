import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
import scipy.io
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class StreetHazards(VisionDataset):
    """
    Benchmark Dataset for Anomaly Segmentation.

    From the paper *Scaling Out-of-Distribution Detection for Real-World Settings*

    :see Paper: https://arxiv.org/pdf/1911.11132.pdf
    :see Website: https://github.com/hendrycks/anomaly-seg
    """

    base_folder = "dtd/images/"
    url = {
        "test": "https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar",
        "train": "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar",
    }
    filename = {"test": "streethazards_test.tar", "train": "streethazards_train.tar"}
    tgz_md5 = {"test": "8c547c1346b00c21b2483887110bfea7"}

    def __init__(
        self,
        root: str,
        train=True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(StreetHazards, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = []
        for d in os.listdir(self.basedir):
            self.files.extend(
                [join(d, f) for f in os.listdir(join(self.basedir, d)) if not f.startswith(".")]
            )
        log.info(f"Found {len(self.files)} texture files.")

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
