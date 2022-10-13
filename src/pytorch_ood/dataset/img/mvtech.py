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


class MVTechAD(ImageDatasetBase):
    """
    MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.
    The dataset provides segmentation masks for anomalies.

    .. image:: https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/dataset_overview_large.png
        :width: 800px
        :alt: MVTech Anomaly Detection Dataset
        :align: center

    :see Paper: https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf
    :see Download: https://www.mvtec.com/company/research/datasets/mvtec-ad/

    Split must be one of ``train`` or ``test``.

    Subset classes can be one of ``bottle``, ``cable``, ``capsule``, ``carpet``,
    ``grid``, ``hazelnut``, ``leather``, ``metal_nut``, ``pill``, ``screw``, ``tile``,
    ``toothbrush``, ``transistor``, ``wood`` and ``zipper``.
    """

    splits = ["train", "test"]
    subsets = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"

    filename = "mvtec_anomaly_detection.tar.xz"

    tgz_md5s = "4b34b33045869ee6d424616cd3a65da3"

    def __init__(
        self,
        root: str,
        split: str,
        subset: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root directory
        :param split: split directory
        :param subset: subset class to use
        :param transform: transformations to apply to image
        :param target_transform: transformation to apply to target masks
        :param download: set to true to automatically download the dataset
        """
        super(ImageDatasetBase, self).__init__(
            join(root, "mvtech-ad"),
            transform=transform,
            target_transform=target_transform,
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

        if subset:
            if subset in self.subsets:
                self.subset = subset
            else:
                raise ValueError(f"Invalid subset: {subset}, possible values: {self.subset}")
        else:
            self.subset = None

        self.files = None
        self.labels = None

        self.load()

    def _get_subset_files(self, subset_dir):
        """
        Returns two lists with filenames to images and corresponding segmentation masks.
        For instances without anomalies, the segmentation mask file will be None.
        """
        ls = []
        fs = []

        defect_dirs = os.listdir(join(subset_dir, self.split))
        for defect_dir in defect_dirs:
            files = glb(join(subset_dir, self.split, defect_dir, "*.png"))
            files.sort()  # sort, since glob does not guarantee ordering

            if defect_dir == "good":
                labels = [None] * len(files)
            else:
                labels = glb(join(subset_dir, "ground_truth", defect_dir, "*_mask.png"))
                labels.sort()

            ls += labels
            fs += files

        return fs, ls

    def _get_all_files(self, root):
        files = list()
        labels = list()
        # Iterate over all the the subsets
        for subset in self.subsets:
            # Create full path
            subset_dir = join(root, subset)
            if os.path.isdir(subset_dir):
                # Iterate over the folders in subset
                img_paths, mask_paths = self._get_subset_files(subset_dir)
                files += img_paths
                labels += mask_paths

        return files, labels

    def load(self):
        if self.subset:
            self.files, self.labels = self._get_subset_files(join(self.root, self.subset))
        else:
            self.files, self.labels = self._get_all_files(self.root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: index
        :returns: (image, target) where target is the segmentation mask
        """
        img_path = self.files[index]
        target = self.labels[index]

        img = Image.open(img_path)

        if target is None:
            target = torch.zeros(size=img.size)
        else:
            target = -1 * torch.tensor(np.array(Image.open(target)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
