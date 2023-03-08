import logging
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class CIFAR10C(ImageDatasetBase):
    """
    Corrupted version of the CIFAR10 from the paper *Benchmarking Neural
    Network Robustness to Common Corruptions and Perturbations.*

    :see Website: `Zenodo <https://zenodo.org/record/2535967>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1903.12261>`__
    """

    subsets = [
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
    Corrupted version of the CIFAR100 from the paper *Benchmarking Neural Network
    Robustness to Common Corruptions and Perturbations.*

    :see Website: `Zenodo <https://zenodo.org/record/3555552>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1903.12261>`__
    """

    base_folder = "CIFAR-100-C/"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
    md5hash = "11f0ed0f1191edbf9fa23466ae6021d3 "
