#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cub 200 Dataset Adapter
from https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py

"""
import logging
import os

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

log = logging.getLogger(__name__)


class Cub2011(VisionDataset):
    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        loader=default_loader,
        target_transform=None,
        download=False,
    ):
        self.root = os.path.expanduser(root)
        self.transforms = transform
        self.loader = loader
        self.train = train
        # self.classes = None
        self._targets = None
        self.target_transform = target_transform
        if download:
            self._download()
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

    @property
    def targets(self):
        return list(self._targets)

    @property
    def classes(self):
        return np.unique(self.targets)

    def _load_metadata(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("You have to install pandas to use the cub200 dataset.")
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )
        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
        # classes are labeled with ascending numbers, starting with 1
        # we have to substract 1 to be in out interval [0, classes-1]
        self._targets = np.array(self.data["target"].apply(lambda x: x - 1))

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            log.exception(e)
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False

        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            # print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self._targets[idx]
        img = self.loader(path)
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        assert img is not None
        assert target is not None
        return img, target
