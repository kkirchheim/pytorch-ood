"""
Stanford cars dataset

http://ai.stanford.edu/~jkrause/cars/car_dataset.html
"""

import logging
import os

import numpy as np
import scipy.io
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

log = logging.getLogger(__name__)


class StanfordCars(VisionDataset):
    def __init__(
        self,
        root,
        train=True,
        loader=default_loader,
        target_transform=None,
        transform=None,
        download=False,
    ):
        self.root = root
        self.train = train
        self.base_name = "cars"
        self.mat_filename = "cars_annos.mat"
        self.targets = None
        self.files = None
        self.loader = loader
        self.target_transform = target_transform
        self.transform = transform
        if download:
            log.warning("Donwloading is not implemented")
        self._load_metadata()

    def _load_metadata(self):
        mat_path = os.path.join(self.root, self.base_name, self.mat_filename)
        self.mat = scipy.io.loadmat(mat_path)
        annotations = self.mat["annotations"][0]
        # select relevant indexes
        sample_is_train = np.squeeze(np.array([1 - i[6] for i in annotations]))
        idx = sample_is_train == self.train
        annotations = annotations[idx]
        self.targets = np.squeeze(np.array([i[5] - 1 for i in annotations]))
        self.files = np.array([i[0][0] for i in annotations])

    @property
    def classes(self):
        return np.unique(self.targets)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        path = os.path.join(self.root, self.base_name, filepath)
        target = self.targets[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
