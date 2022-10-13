import logging
import os
import string
from os.path import join
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets.utils import check_integrity, download_url

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


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

    char_list = list(map(str, string.digits + string.ascii_uppercase + string.ascii_lowercase))

    def _get_file_list(self, dirName):
        listOfFile = os.listdir(dirName)
        allFiles = list()
        allLabels = list()

        for entry in listOfFile:
            fullPath = os.path.join(dirName, entry)
            label = int(str(Path(fullPath).name).split("Sample")[1]) - 1
            label = ord(self.char_list[label])
            Files = os.listdir(fullPath)
            for files in Files:
                allFiles.append(os.path.join(fullPath, files))
                allLabels.append(label)
        return allFiles, allLabels

    def _get_all_files(self, dirName):
        files_set_1, labels_set_1 = self._get_file_list(join(dirName, "GoodImg", "Bmp"))

        return files_set_1, labels_set_1

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

        self._load()

    def _check_integrity(self) -> bool:
        return check_integrity(join(self.root, self.filename_dataset), self.tgz_dataset_md5)

    def _download(self):
        import tarfile

        if self._check_integrity():
            return

        download_url(self.url_dataset, self.root, self.filename_dataset, self.tgz_dataset_md5)
        download_url(self.url_list, self.root, self.filename_list, self.tgz_list_md5)
        with tarfile.open(os.path.join(self.root, self.filename_dataset), "r:gz") as tar:
            tar.extractall(path=self.root)
        with tarfile.open(os.path.join(self.root, self.filename_list), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load(self):
        self.files, self.labels = self._get_all_files(join(self.root, "English", "Img"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is ascii value of the image class.
        """
        img_path = self.files[index]
        target = self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
