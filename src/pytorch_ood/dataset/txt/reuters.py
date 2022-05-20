"""
Much of the code is taken from the baseline-implementation:
https://github.com/hendrycks/error-detection/blob/master/NLP/Categorization/Reuters52.ipynb
"""
import logging
import os
import re
from typing import Tuple

import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from torchvision.datasets.utils import download_url

from .stop_words import stop_words

log = logging.getLogger(__name__)


class Reuters52(Dataset):
    """
    Stemmed version of the reuters 52 dataset, as used by Hendrycks et al.
    """

    train_url = "https://raw.githubusercontent.com/hendrycks/error-detection/master/NLP/Categorization/data/r52-train.txt"
    test_url = "https://raw.githubusercontent.com/hendrycks/error-detection/master/NLP/Categorization/data/r52-test.txt"
    test_md5 = "8a82cdf79e111df1bb23a9bbc48f6d25"
    train_md5 = "6b1d32bd95e95c1c26cd592d3bdb8c0e"
    train_filename = "r52-train-stemmed.txt"
    test_filename = "r52-test-stemmed.txt"
    class2index = {
        "acq": 0,
        "alum": 1,
        "bop": 2,
        "carcass": 3,
        "cocoa": 4,
        "coffee": 5,
        "copper": 6,
        "cotton": 7,
        "cpi": 8,
        "cpu": 9,
        "crude": 10,
        "dlr": 11,
        "earn": 12,
        "fuel": 13,
        "gas": 14,
        "gnp": 15,
        "gold": 16,
        "grain": 17,
        "heat": 18,
        "housing": 19,
        "income": 20,
        "instal-debt": 21,
        "interest": 22,
        "ipi": 23,
        "iron-steel": 24,
        "jet": 25,
        "jobs": 26,
        "lead": 27,
        "lei": 28,
        "livestock": 29,
        "lumber": 30,
        "meal-feed": 31,
        "money-fx": 32,
        "money-supply": 33,
        "nat-gas": 34,
        "nickel": 35,
        "orange": 36,
        "pet-chem": 37,
        "platinum": 38,
        "potato": 39,
        "reserves": 40,
        "retail": 41,
        "rubber": 42,
        "ship": 43,
        "strategic-metal": 44,
        "sugar": 45,
        "tea": 46,
        "tin": 47,
        "trade": 48,
        "veg-oil": 49,
        "wpi": 50,
        "zinc": 51,
    }

    def __init__(self, root, transform=None, target_transform=None, train=True, download=True):
        """

        :param root:
        :param transform:
        :param target_transform:
        :param train:
        :param download:
        """
        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.target_transform = target_transform
        self.is_train = train
        self._targets = []
        self._analyzer = None
        if download:
            self._download()
        self._labels, self._data = self._load_data()
        # mapping class names to integers
        for i, label in enumerate(self._labels):
            for clazz, index in self.class2index.items():
                if label.startswith(clazz):
                    self._targets.append(index)
        self._targets = np.array(self._targets)

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
        x, targets = [], []
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                text = " ".join(word for word in words[2:] if word not in stop_words)
                x.append(text)
                targets.append(".".join(words[0:2]))
        return targets, x

    def _check_integrity(self):
        try:
            self._load_data()
        except Exception as e:
            # log.exception(e)
            return False

        return True

    def __getitem__(self, index):
        x = self._data[index]
        y = self._targets[index]
        if self.target_transform:
            y = self.target_transform(y)
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self._data)


class Reuters8(Reuters52):
    """
    Stemmed version of the reuters 8 dataset, as used by Hendrycks et al.
    """

    train_url = "https://raw.githubusercontent.com/hendrycks/error-detection/master/NLP/Categorization/data/r8-train.txt"
    test_url = "https://raw.githubusercontent.com/hendrycks/error-detection/master/NLP/Categorization/data/r8-test.txt"
    test_md5 = "7a54d01272de570d13d0c58cf8aa3c8d"
    train_md5 = "c979a285f1c132c5be8d554b385f2c49"
    train_filename = "r8-train-stemmed.txt"
    test_filename = "r8-test-stemmed.txt"

    def __init__(self, root, transform=None, target_transform=None, train=True, download=True):
        """

        :param root:
        :param transform:
        :param target_transform:
        :param train:
        :param download:
        """
        super(Reuters52, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.target_transform = target_transform
        self.is_train = train
        self._analyzer = None
        if download:
            self._download()
        self._targets, self._data = self._load_data()

    def _load_data(self) -> Tuple:
        if self.is_train:
            filename = self.train_filename
        else:
            filename = self.test_filename
        filename = os.path.join(self.root, filename)
        x, targets = [], []
        with open(filename, "r") as f:
            for line in f:
                line = re.sub(r"\W+", " ", line).strip()
                x.append(line[1:])
                x[-1] = " ".join(word for word in x[-1].split() if word not in stop_words)
                targets.append(line[0])
        return np.array(targets, dtype=int), x
