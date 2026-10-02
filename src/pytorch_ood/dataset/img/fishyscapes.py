import logging
import os
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_md5, download_and_extract_archive

from ...api import DatasetInfo, Paper, Role, Task

log = logging.getLogger(__name__)


class FishyScapes(Dataset):
    """
    The FishyScapes dataset contains images from the CityScapes dataset blended with unknown objects
    scraped from the web.
    You additionally have to manually download the CityScapes validation dataset (left, 8 bit).

    The dataset contains annotations for a *void*-class that should be ignored during evaluation.
    Images are :class:`PIL.Image.Image` of size :math:`2048 \\times 1024`. The target is a long tensor of shape
    :math:`H \\times W` with value ``0`` for in-distribution pixels, ``-1`` for anomalous pixels and
    ``VOID_LABEL`` (``1``) for void pixels.

    There are currently three versions:

    * ``1.0.0`` - not blended
    * ``2.0.0`` - slightly blended
    * ``3.0.0`` - well blended

    .. image:: https://fishyscapes.com/assets/img/example3.jpg
        :width: 800px
        :alt: FishyScapes example
        :align: center
    """

    info = DatasetInfo(
        task=Task.SEGMENTATION,
        roles={Role.BENCHMARK},
        license="CC-BY-4.0 (Lost and Found validation annotations)",
        paper=Paper(
            title="The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation",
            venue="IJCV",
            year=2021,
            url="https://arxiv.org/abs/1904.03215",
            code="https://github.com/hermannsblum/bdl-benchmark",
        ),
        homepage="https://fishyscapes.com/",
    )

    dataset_links = {
        "1.0.0": (
            "http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fs_val_v1.zip",
            "fs_val_v1.zip",
            "79fb134419c83f2f20b575955efa9d20",
        ),
        "2.0.0": (
            "http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fs_val_v2.zip",
            "fs_val_v2.zip",
            "5088c63497927200d935c41d54b1cb23",
        ),
        "3.0.0": (
            "http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fs_val_v3.zip",
            "fs_val_v3.zip",
            "0dc11db9e57088c5bb18de4c55a53f3a",
        ),
    }

    VOID_LABEL = 1  #: void label, should be ignored during score calculation

    def __init__(self, root, cs_root, version="3.0.0", download: bool = False, transforms=None):
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param cs_root: path to the CityScapes ``leftImg8bit/val`` directory (with one sub-directory per city),
            which has to be downloaded manually
        :param version: can be one of ``1.0.0``, ``2.0.0``, ``3.0.0``
        :param download: download the data to ``root`` if it is not found there
        :param transforms: called as ``transforms(image, mask)`` with the PIL image and the target mask,
            and must return the transformed ``(image, mask)`` tuple
        :raises AssertionError: if ``version`` is unknown
        """
        assert version in self.dataset_links.keys(), f"Unknown dataset version: '{version}'"

        self.root = root
        self.cs_root = cs_root
        self.transforms = transforms
        self.version = version
        self.dirname = f"fishyscapes-{version}"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.files = [
            f for f in sorted(os.listdir(join(self.root, self.dirname))) if f.endswith("_rgb.npz")
        ]

    def _check_integrity(self):
        url, filename, md5hash = self.dataset_links[self.version]
        if not os.path.exists(join(self.root, filename)):
            return False

        return check_md5(join(self.root, filename), md5hash)

    def download(self):
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        url, filename, md5hash = self.dataset_links[self.version]
        download_and_extract_archive(
            url,
            self.root,
            md5=md5hash,
            filename=filename,
            extract_root=join(self.root, self.dirname),
        )

    def _get_org_img(self, path):
        """
        0000_frankfurt_000001_046504_rgb.npz -> frankfurt/frankfurt_000001_046504_leftImg8bit.png
        """
        parts = path.split("_")
        city = parts[1]
        path = os.path.join(city, "_".join(parts[1:]))
        return path.replace("rgb.npz", "leftImg8bit.png")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        overlay_path = self.files[index]
        orig_path = self._get_org_img(overlay_path)

        overlay_path = join(self.root, self.dirname, overlay_path)
        orig_path = join(self.cs_root, orig_path)

        orig = np.array(Image.open(orig_path)).astype(int)
        overlay = np.load(overlay_path)["rgb"].astype(int)

        void_label_path = overlay_path.replace("_rgb.npz", "_labels.png")
        void_labels = np.array(Image.open(void_label_path)).astype(int)

        mask = np.where(np.where(overlay != 0, True, False).any(axis=2), -1, 0)
        mask[np.logical_and(void_labels != 0, mask >= 0)] = self.VOID_LABEL

        img = np.clip(orig + overlay, 0, 255).astype(np.uint8)

        img = Image.fromarray(img)
        mask = torch.tensor(mask).long()

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask


class LostAndFound(Dataset):
    """
    The LostAndFound dataset contains images from driving scenarios with real world anomalies.
    It can be used with models trained on CityScapes.

    The dataset contains annotations for a *void*-class that should be ignored during evaluation.
    The labels are provided by FishyScapes.
    Images are :class:`PIL.Image.Image`. The target is an integer tensor of shape :math:`H \\times W` with value ``-1``
    for anomalous pixels, ``VOID_LABEL`` (``1``) for void pixels and ``0`` for in-distribution pixels.

    .. image:: https://fishyscapes.com/assets/img/laf_0008_rgb.jpg
        :width: 800px
        :alt: LostAndFound (Fishy edition) example
        :align: center

    .. warning:: The image with index 79 does not contain any outlier pixels.
    """

    info = DatasetInfo(
        task=Task.SEGMENTATION,
        roles={Role.BENCHMARK},
        license="Daimler AG non-commercial license",
        paper=Paper(
            title="Lost and Found: Detecting Small Road Hazards for Self-Driving Vehicles",
            venue="IROS",
            year=2016,
            url="https://arxiv.org/abs/1609.04653",
        ),
        homepage="http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html",
    )

    annotation_url = (
        "http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fishyscapes_lostandfound.zip",
        "fishyscapes_lostandfound.zip",
        "0d3bf7c0ec38bd50b84f3d8aaa4b2e26",
    )

    annotation_dir = "fishyscapes_lostandfound"

    data_url = (
        "http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/leftImg8bit.zip",
        "leftImg8bit.zip",
        "08eaa79ce05126f6bd22a3ca563746d0",
    )

    data_dir = join("lostandfound", "leftImg8bit")

    VOID_LABEL = 1  #: void label, should be ignored during score calculation

    def __init__(self, root, download=False, transforms=None):
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param download: download the data to ``root`` if it is not found there
        :param transforms: called as ``transforms(image, mask)`` with the PIL image and the target mask,
            and must return the transformed ``(image, mask)`` tuple
        """
        self.root = root
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.ano_files = sorted(os.listdir(join(self.root, self.annotation_dir)))

    def _check_integrity(self):
        url, filename, md5hash = self.annotation_url
        if not os.path.exists(join(self.root, filename)):
            return False

        if not check_md5(join(self.root, filename), md5hash):
            return False

        url, filename, md5hash = self.data_url

        if not os.path.exists(join(self.root, filename)):
            return False

        if not check_md5(join(self.root, filename), md5hash):
            return False

        return True

    def download(self):
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        url, filename, md5hash = self.annotation_url
        download_and_extract_archive(
            url,
            self.root,
            md5=md5hash,
            filename=filename,
            extract_root=join(self.root, self.annotation_dir),
        )

        url, filename, md5hash = self.data_url
        download_and_extract_archive(
            url,
            self.root,
            md5=md5hash,
            filename=filename,
            extract_root=join(self.root, "lostandfound"),
        )

    def _get_org_img(self, ano_path):
        """
        0000_04_Maurener_Weg_8_000000_000030_labels.png -> 04_Maurener_Weg_8/04_Maurener_Weg_8_000000_000030_leftImg8bit.png
        """

        parts = ano_path.split("_")
        # discard the last three parts
        direc = "_".join(parts[1:-3])
        file = "_".join(parts[1:]).replace("_labels.png", "_leftImg8bit.png")
        return join(direc, file)

    def __len__(self):
        return len(self.ano_files)

    def __getitem__(self, index):
        ano_path = self.ano_files[index]
        img_path = self._get_org_img(ano_path)

        img_path_abs = join(self.root, self.data_dir, "train", img_path)
        if not os.path.exists(img_path_abs):
            img_path_abs = join(self.root, self.data_dir, "test", img_path)

        ano_path = join(self.root, self.annotation_dir, ano_path)

        img = Image.open(img_path_abs)
        targets = np.array(Image.open(ano_path), dtype=np.int32)

        targets = np.where(targets == 1, -1, targets)
        targets = np.where(targets == 255, self.VOID_LABEL, targets)

        targets = torch.tensor(targets)

        if self.transforms:
            img, targets = self.transforms(img, targets)

        return img, targets
