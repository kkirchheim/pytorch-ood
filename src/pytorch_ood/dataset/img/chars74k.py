import logging
import os
import tarfile
from os.path import join
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_url

from pytorch_ood.dataset.img.base import ImageDatasetBase

log = logging.getLogger(__name__)


def _get_file_list(dir_name):
    file_list = os.listdir(dir_name)
    all_files = list()
    all_labels = list()

    for entry in file_list:
        full_path = os.path.join(dir_name, entry)
        label = int(str(Path(full_path).name).split("Sample")[1]) - 1
        dir_files = os.listdir(full_path)
        for files in dir_files:
            all_files.append(os.path.join(full_path, files))
            all_labels.append(label)
    return all_files, all_labels


class Chars74k(ImageDatasetBase):
    """
    Dataset from the paper *Character Recognition in Natural Images*. Can be used as example OOD data.

    .. image:: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/Samples/confusing_english.png
        :width: 800px
        :alt: Chars47k Dataset Example
        :align: center

    :see Website: `Link <http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/>`__
    :see Paper: `Link <http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf>`__
    """

    base_folder = "chars74k"
    url_dataset = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz"
    url_list = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ListsTXT.tgz"
    filename_dataset = "EnglishImg.tgz"
    filename_list = "ListsTXT.tgz"

    tgz_dataset_md5 = "85d157e0c58f998e1cda8def62bcda0d"
    tgz_list_md5 = "7d7b8038b3c47bf2a1c5a80c1dd79a0d"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """
        :param root: root directory of dataset
        :param transform: transformation to apply to the images
        :param target_transform: transformation to apply to the labels
        :param download: set to true to automatically download the dataset
        """
        super(ImageDatasetBase, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.root = join(root, self.base_folder)

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.class_mapping = {}
        self._load()
        self._create_class_remapping()

    def _check_integrity(self) -> bool:
        return check_integrity(join(self.root, self.filename_dataset), self.tgz_dataset_md5)

    def _download(self):
        if self._check_integrity():
            return

        download_url(self.url_dataset, self.root, self.filename_dataset, self.tgz_dataset_md5)
        download_url(self.url_list, self.root, self.filename_list, self.tgz_list_md5)
        with tarfile.open(os.path.join(self.root, self.filename_dataset), "r:gz") as tar:
            tar.extractall(path=self.root)
        with tarfile.open(os.path.join(self.root, self.filename_list), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load(self):
        self.files, self.labels = _get_file_list(
            join(self.root, "English", "Img", "GoodImg", "Bmp")
        )

    def _create_class_remapping(self):
        """
        For remapping labels into range 0, ..., n_classes - 1
        """
        labels = np.unique(np.array(self.labels))
        self.class_mapping = {k: v for k, v in zip(labels, range(len(labels)))}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        target = self.labels[index]

        target = self.class_mapping[target]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
