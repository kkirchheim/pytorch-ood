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

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class StreetHazards(ImageDatasetBase):
    """
    Benchmark Dataset for Anomaly Segmentation.

    From the paper *Scaling Out-of-Distribution Detection for Real-World Settings*

    :see Paper: https://arxiv.org/pdf/1911.11132.pdf
    :see Website: https://github.com/hendrycks/anomaly-seg
    """

    base_folder = "dtd/images/"

    subset_list = ["test", "train", "validation"]

    base_folder_list = [
        "test/images/",
        "train/images/training/",
        "train/images/validation/",
    ]
    url_list = [
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar",
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar",
        "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar",
    ]
    filename_list = [
        "streethazards_test.tar",
        "streethazards_train.tar",
        "streethazards_train.tar",
    ]
    tgz_md5_list = [
        "8c547c1346b00c21b2483887110bfea7",
        "cd2d1a8649848afb85b5059d227d2090",
        "cd2d1a8649848afb85b5059d227d2090",
    ]

    def getListOfFiles(self, dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)

        return allFiles

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root path for dataset
        :param subset: one of 'train', 'test', 'validation'
        :param transform: transformations to apply to images
        :param target_transform: transformations to apply to target
        :param download: if dataset should be downloaded automatically
        """
        super(StreetHazards, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        if download:
            self.download()

        if subset not in self.subset_list:
            raise ValueError(f"Invalid subset: {subset}")

        self.base_folder = self.base_folder_list[self.subset_list.index(subset)]
        self.url = self.url_list[self.subset_list.index(subset)]
        self.filename = self.filename_list[self.subset_list.index(subset)]
        self.tgz_md5 = self.tgz_md5_list[self.subset_list.index(subset)]

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

        self.files = self.getListOfFiles(self.basedir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is annotation of the image.
        """

        file, target = self.files[index], self.files[index].replace("images", "annotations")

        # to return a PIL Image
        img = Image.open(file)
        target = Image.open(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
