"""
"""
import logging
import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

log = logging.getLogger(__name__)


class WikiText2(Dataset):
    """
    WikiText2 dataset
    Contains collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia

    :see Website: https://arxiv.org/abs/1609.07843
    """

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    md5 = "542ccefacc6c27f945fb54453812b3cd"
    base_dir = "wikitext-2"
    filenames = {
        "train": "wiki.train.tokens",
        "test": "wiki.test.tokens",
        "val": "wiki.valid.tokens",
    }

    def __init__(self, root, split, transform=None, target_transform=None, download=True):
        if split not in list(self.filenames.keys()):
            raise ValueError(f"Invalid split: {split}")

        super(Dataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.target_transform = target_transform
        self.split = split

        if download:
            self._download()

        self._data = self._load_data()

    def _download(self):
        if self._check_integrity():
            log.info("Files already downloaded and verified")
            return

        download_and_extract_archive(
            url=self.url, download_root=self.root, extract_root=self.root, md5=self.md5
        )

    def _load_data(self) -> Tuple:
        filename = self.filenames[self.split]

        filename = os.path.join(self.root, self.base_dir, filename)
        x = []
        with open(filename, "r", encoding="utf8") as f:
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


class WikiText103(WikiText2):
    """
    WikiText103 dataset
    Contains collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia

    :see Website: https://arxiv.org/abs/1609.07843
    """

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    md5 = "9ddaacaf6af0710eda8c456decff7832"
    base_dir = "wikitext-103"
    filenames = {
        "train": "wiki.train.tokens",
        "test": "wiki.test.tokens",
        "val": "wiki.valid.tokens",
    }
