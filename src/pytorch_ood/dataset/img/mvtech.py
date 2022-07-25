import glob
import logging
import os
from glob import glob as glb
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class MVTECH(ImageDatasetBase):
    """
    MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.

    :see Paper: https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf
    :see Download: https://www.mvtec.com/company/research/datasets/mvtec-ad/
    """

    splits = ["train", "test"]

    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"

    filename = "mvtec_anomaly_detection.tar.xz"

    tgz_md5s = "4b34b33045869ee6d424616cd3a65da3"

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root directory
        :param split: split directory
        :param transform: transformations to apply to image
        :param target_transform: transformation to apply to target masks
        :param download: set to true to automatically download the dataset
        """
        super(ImageDatasetBase, self).__init__(
            join(root, "MVTECH-AD"), transform=transform, target_transform=target_transform
        )

        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")
        else:
            self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.load()

    def _get_subset_files(self, subset_dir):
        ls = []
        fs = []

        folders = os.listdir(join(subset_dir, self.split))
        for folder in folders:
            files = glb(join(subset_dir, self.split, folder, "*.png"))
            files.sort()

            if folder == "good":
                labels = [None] * len(files)
            else:
                labels = glb(join(subset_dir, "ground_truth", folder, "*_mask.png"))
                labels.sort()

            # for f, l in zip(files, labels):
            #     print(f"{os.path.basename(f)} -> {os.path.basename(l) if l is not None else l}")

            ls += labels
            fs += files

        return fs, ls

    def get_all_files(self, root):
        subsets = os.listdir(root)
        files = list()
        labels = list()
        # Iterate over all the the subsets
        for subset in subsets:
            # Create full path
            subset_dir = join(root, subset)
            if os.path.isdir(subset_dir):
                # Iterate over the folders in subset
                img_paths, mask_paths = self._get_subset_files(subset_dir)
                files += img_paths
                labels += mask_paths

        return files, labels

    def load(self):
        self.files, self.labels = self.get_all_files(self.root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.files[index]
        target = self.labels[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.open(img_path)

        if target is None:
            target = torch.zeros(size=img.size)
        else:
            target = torch.tensor(np.array(Image.open(target)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
