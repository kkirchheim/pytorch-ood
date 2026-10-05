"""
Much of the code is taken from the baseline-implementation:
https://github.com/hendrycks/outlier-exposure/blob/master/NLP_classification/multi30k/
"""

import logging
import os
from typing import Any, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from ...api import DatasetInfo, Task

log = logging.getLogger(__name__)


class Multi30k(Dataset):
    """
    Multi-30k dataset, as used by Hendrycks et al.

    Usually used as OOD data, labels are -1 by default. Each item is a tuple ``(text, -1)`` with the sentence
    as :class:`str`.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        license=None,
    )

    train_url = "https://github.com/hendrycks/outlier-exposure/raw/master/NLP_classification/multi30k/train.txt"
    test_url = "https://raw.githubusercontent.com/hendrycks/outlier-exposure/master/NLP_classification/multi30k/val.txt"

    test_md5 = "8d407ae05dbc61e3e61ffd3a3f9d39fb"
    train_md5 = "094c68794632983239bf1bc1b37282a8"
    train_filename = "m30k-train.txt"
    test_filename = "m30k-test.txt"

    def __init__(self, root, transform=None, target_transform=None, train=True, download=False):
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param transform: function applied to the text (a :class:`str`)
        :param target_transform: function applied to the target
        :param train: use the training file (``True``) or the validation file ``val.txt`` (``False``)
        :param download: download the data to ``root`` if it is not found there
        """
        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.target_transform = target_transform
        self.is_train = train

        if download:
            self._download()

        self._data = self._load_data()

    def _download(self):
        if self._check_integrity():
            log.info("Files already downloaded and verified")
            return

        if self.is_train:
            filename = self.train_filename
            md5 = self.train_md5
            url = self.train_url
        else:
            filename = self.test_filename
            md5 = self.test_md5
            url = self.test_url
        download_url(url, self.root, filename, md5)

    def _load_data(self) -> Tuple:
        if self.is_train:
            filename = self.train_filename
        else:
            filename = self.test_filename
        filename = os.path.join(self.root, filename)
        x = []
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                text = " ".join(word for word in words)
                x.append(text)

        return x

    def _check_integrity(self):
        try:
            self._load_data()
        except Exception as e:
            # log.exception(e)
            return False

        return True

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: index of the sample
        :return: tuple ``(text, target)`` of the text as :class:`str` (or the output of ``transform``)
            and the target ``-1`` (or the output of ``target_transform``)
        """
        x = self._data[index]
        y = -1

        if self.target_transform:
            y = self.target_transform(y)
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self._data)
