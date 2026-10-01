import logging
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from ...api import DatasetInfo, Paper, Role, Task
from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class CIFAR10C(ImageDatasetBase):
    """
    Corrupted version of the CIFAR10 test set from the paper *Benchmarking Neural
    Network Robustness to Common Corruptions and Perturbations.*

    Images are returned as :class:`PIL.Image.Image` of size :math:`32 \\times 32`. Targets are the original class
    labels, **not** ``-1``. Each corruption is available at 5 severity levels, which are concatenated in
    the order of the severity (10,000 images each). The corruptions are ``brightness``, ``contrast``,
    ``defocus_blur``, ``elastic_transform``, ``fog``, ``frost``, ``gaussian_blur``, ``gaussian_noise``,
    ``glass_blur``, ``impulse_noise``, ``jpeg_compression``, ``motion_blur``, ``pixelate``, ``saturate``,
    ``shot_noise``, ``snow``, ``spatter``, ``speckle_noise`` and ``zoom_blur``.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license="CC-BY-4.0",
        paper=Paper(
            title="Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
            venue="ICLR",
            year=2019,
            url="https://arxiv.org/abs/1903.12261",
        ),
        homepage="https://zenodo.org/record/2535967",
    )

    subsets = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ]

    base_folder = "CIFAR-10-C"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"
    md5hash = "56bf5dcef84df0e2308c6dcbcbbd8499"

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param subset: corruption to load, see above, or ``all`` to concatenate all corruptions (the labels are
            repeated accordingly)
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target
        :param download: download the data to ``root`` if it is not found there
        :raises ValueError: if ``subset`` is unknown
        """
        super(CIFAR10C, self).__init__(root, transform, target_transform, download)

        self.subset = subset

        if subset not in self.subsets and subset != "all":
            raise ValueError(f"Unknown Subset: {subset}")

        if subset == "all":
            self.data = np.concatenate(
                [np.load(join(root, self.base_folder, f"{s}.npy")) for s in self.subsets]
            )
            self.targets = np.concatenate(
                [np.load(join(root, self.base_folder, "labels.npy")) for s in self.subsets]
            )
        else:
            self.data = np.load(join(root, self.base_folder, f"{subset}.npy"))
            self.targets = np.load(join(root, self.base_folder, "labels.npy"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.data[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100C(CIFAR10C):
    """
    Corrupted version of the CIFAR100 test set from the paper *Benchmarking Neural Network
    Robustness to Common Corruptions and Perturbations.* Same format and corruptions as :class:`CIFAR10C`,
    with the CIFAR100 class labels as targets.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license="CC-BY-4.0",
        paper=Paper(
            title="Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
            venue="ICLR",
            year=2019,
            url="https://arxiv.org/abs/1903.12261",
        ),
        homepage="https://zenodo.org/record/3555552",
    )

    base_folder = "CIFAR-100-C/"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
    md5hash = "11f0ed0f1191edbf9fa23466ae6021d3"
