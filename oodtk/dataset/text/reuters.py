import logging
import os
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.utils import download_url

from oodtk.dataset import OSRDataset

from oodtk.dataset.text.stop_words import stop_words

log = logging.getLogger(__name__)


class Reuters52(Dataset):
    def __init__(self, root, download=False):
        super(Reuters52, self).__init__()
        self._dataset1 = Reuters52Base(root, train=True, download=download)
        self._dataset2 = Reuters52Base(root, train=False, download=download)
        self.dataset = ConcatDataset([self._dataset1, self._dataset2])

    def __getitem__(self, item):
        x, y = self.dataset[item]

        if self.target_transform:
            y = self.target_transform(y)

        if self.transforms:
            x = self.transforms(x)

        return x, y

    def __len__(self):
        return len(self.dataset)

    def unique_targets(self) -> np.ndarray:
        return np.unique(np.concatenate(self._dataset1._labels, self._dataset2._labels))


class Reuters52Base(Dataset):
    """
    Stemmed version of the reuters 52 dataset.

    Much of the code is taken from the baseline-implementation:
    https://github.com/hendrycks/error-detection/blob/master/NLP/Categorization/Reuters52.ipynb
    """

    train_url = (
        "https://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-stemmed.txt"
    )
    test_url = (
        "https://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-stemmed.txt"
    )

    test_md5 = "e29d9f65d622f926dee08a6a87b2277a"
    train_md5 = "f115a957fbbf050a81d38ca08500e5c3"

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

    def __init__(
        self, root, transform=None, target_transform=None, train=True, download=True
    ):
        """
        TODO: add support for custom loader?
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


if __name__ == "__main__":
    root = os.path.expanduser(os.path.join("~", ".cache", "oodtk"))
    dataset = Reuters52(root, download=True)
    pass
