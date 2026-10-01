""" """

import logging
import os
from typing import Any, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from ...api import DatasetInfo, Paper, Task

log = logging.getLogger(__name__)


class WikiText2(Dataset):
    """
    Contains a collection of about 2 million tokens extracted from the set of verified Good and
    Featured articles on Wikipedia.

    Usually used as OOD (training) data, for example, for
    :class:`Outlier Exposure <pytorch_ood.loss.OutlierExposureLoss>`. Labels are -1 by default.
    Each item is a tuple ``(text, -1)``, where ``text`` is one line of the token file as a :class:`str`
    (including empty lines and headings).
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        license="CC-BY-SA-3.0",
        paper=Paper(
            title="Pointer Sentinel Mixture Models",
            venue="ICLR",
            year=2017,
            url="https://arxiv.org/abs/1609.07843",
        ),
    )

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    md5 = "542ccefacc6c27f945fb54453812b3cd"
    base_dir = "wikitext-2"
    filenames = {
        "train": "wiki.train.tokens",
        "test": "wiki.test.tokens",
        "val": "wiki.valid.tokens",
    }

    def __init__(self, root, split, transform=None, target_transform=None, download=False):
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param split: one of ``train``, ``test`` and ``val``
        :param transform: function applied to the text (a :class:`str`)
        :param target_transform: function applied to the target
        :param download: download the data to ``root`` if it is not found there
        """
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


class WikiText103(WikiText2):
    """
    Contains a collection of over 100 million tokens extracted from the set of verified Good and Featured
    articles on Wikipedia.

    Usually used as OOD (training) data, for example, for
    :class:`Outlier Exposure <pytorch_ood.loss.OutlierExposureLoss>`. Labels are -1 by default.
    Each item is a tuple ``(text, -1)``, where ``text`` is one line of the token file as a :class:`str`
    (including empty lines and headings).
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        license="CC-BY-SA-3.0",
        paper=Paper(
            title="Pointer Sentinel Mixture Models",
            venue="ICLR",
            year=2017,
            url="https://arxiv.org/abs/1609.07843",
        ),
    )

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    md5 = "9ddaacaf6af0710eda8c456decff7832"
    base_dir = "wikitext-103"
    filenames = {
        "train": "wiki.train.tokens",
        "test": "wiki.test.tokens",
        "val": "wiki.valid.tokens",
    }
