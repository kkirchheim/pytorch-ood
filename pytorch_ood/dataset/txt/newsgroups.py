"""
Much of the code is taken from the baseline-implementation:
https://github.com/hendrycks/error-detection/blob/master/NLP/Categorization/20%20Newsgroups.ipynb
"""
import logging
import os
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from .stop_words import stop_words

log = logging.getLogger(__name__)


class NewsGroup20(Dataset):
    """
    Stemmed etc. version of the newsgroup dataset, as used by Hendrycks et al.
    """

    train_url = (
        "https://raw.githubusercontent.com/hendrycks/outlier-exposure/"
        "master/NLP_classification/20newsgroups/orig_data/20ng-train-no-short.txt"
    )
    test_url = (
        "https://raw.githubusercontent.com/hendrycks/outlier-exposure/"
        "master/NLP_classification/20newsgroups/orig_data/20ng-test-no-short.txt"
    )
    test_md5 = "978c4d8fd3bde6a12a1e8312f0031815"
    train_md5 = "4444a088dda968b44f7a6dec756698b3"
    train_filename = "20ng-train-no-short.txt"
    test_filename = "20ng-test-no-short.txt"
    class_names = [
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ]

    def __init__(self, root, transform=None, target_transform=None, train=True, download=True):
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
        self.class_map = {clazz: index for index, clazz in enumerate(self.class_names)}
        for i, label in enumerate(self._labels):
            for clazz in self.class_names:
                if label.startswith(clazz):
                    self._targets.append(self.class_map[clazz])
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
