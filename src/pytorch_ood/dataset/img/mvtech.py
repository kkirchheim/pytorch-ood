import logging
import os
from glob import glob as glb
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...api import DatasetInfo, Paper, Role, Task
from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class MVTechAD(ImageDatasetBase):
    """
    MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.
    The dataset provides segmentation masks for anomalies. The official name of the dataset is *MVTec AD*;
    this spelling of the class name is kept for backwards compatibility.

    Images are :class:`PIL.Image.Image`. The target is a mask tensor with the size of the image, in which
    non-zero (negative) entries mark anomalous pixels. Images without anomalies get a mask of zeros.

    :see Download: `MVTec AD website <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`__
    """

    info = DatasetInfo(
        task=Task.SEGMENTATION,
        roles={Role.BENCHMARK},
        license="CC-BY-NC-SA-4.0",
        paper=Paper(
            title="The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
            venue="IJCV",
            year=2021,
            url="https://link.springer.com/article/10.1007/s11263-020-01400-4",
        ),
        homepage="https://www.mvtec.com/company/research/datasets/mvtec-ad/",
    )

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
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param split: one of ``train`` or ``test``
        :param subset: object class to use. One of ``bottle``, ``cable``, ``capsule``, ``carpet``, ``grid``,
            ``hazelnut``, ``leather``, ``metal_nut``, ``pill``, ``screw``, ``tile``, ``toothbrush``,
            ``transistor``, ``wood`` and ``zipper``. If ``None``, all classes are used.
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target mask
        :param download: download the data to ``root`` if it is not found there
        :raises ValueError: if ``split`` or ``subset`` is invalid
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

        defect_dirs = sorted(os.listdir(join(subset_dir, self.split)))
        for defect_dir in defect_dirs:
            files = sorted(glb(join(subset_dir, self.split, defect_dir, "*.png")))

            if defect_dir == "good":
                labels = [None] * len(files)
            else:
                labels = sorted(glb(join(subset_dir, "ground_truth", defect_dir, "*_mask.png")))

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
        :return: tuple of the image and the segmentation mask, in which non-zero entries mark anomalous pixels
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
