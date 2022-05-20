"""
Much of the code is taken from the baseline-implementation:
https://github.com/hendrycks/outlier-exposure/blob/master/NLP_classification/wmt16/
"""
import logging
import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

log = logging.getLogger(__name__)


class WMT16Sentences(Dataset):
    """
    WMT16 sentences, as used by Hendrycks et al.
    """

    url = "https://raw.githubusercontent.com/hendrycks/outlier-exposure/master/NLP_classification/wmt16/wmt16_sentences"
    md5 = "6dff65f45ac112c150b8a2cc30509b03"
    filename = "wmt16_sentences"

    def __init__(self, root, transform=None, target_transform=None, download=True):
        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.target_transform = target_transform

        if download:
            self._download()

        self._data = self._load_data()

    def _download(self):
        if self._check_integrity():
            log.info("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.md5)

    def _load_data(self) -> Tuple:
        filename = os.path.join(self.root, self.filename)
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

    def __getitem__(self, index):
        x = self._data[index]
        y = -1

        if self.target_transform:
            y = self.target_transform(y)
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self._data)
