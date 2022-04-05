import logging
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class CIFAR10C(ImageDatasetBase):
    """
    From the paper *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*

    :see Website: https://zenodo.org/record/2535967
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
    tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"

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
        else:
            self.data = np.load(join(root, self.base_folder, f"{subset}.npy"))

        self.targets = np.load(join(root, self.base_folder, "labels.npy"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
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


# class CIFAR10P(ImageDatasetBase):
#     """
#     Natural Adversarial Examples
#
#     :see Website: https://zenodo.org/record/2535967
#     """
#
#     subsets = [
#         "brightness",
#         "gaussian_blur",
#         "gaussian_noise",
#         "gaussian_noise_2",
#         "gaussian_noise_3",
#         "motion_blur",
#         "rotate",
#         "scale",
#         "shear",
#         "shot_noise",
#         "shot_noise_2",
#         "shot_noise_3",
#         "snow",
#         "spatter",
#         "speckle_noise",
#         "speckle_noise_2",
#         "speckle_noise_3",
#         "tilt",
#         "translate",
#         "zoom_blur",
#     ]
#
#     base_folder = "CIFAR-10-P/"
#     url = "https://zenodo.org/record/2535967/files/CIFAR-10-P.tar"
#     filename = "CIFAR-10-P.tar"
#     tgz_md5 = "125d6775afc5846ea84584d7524dedff"
#
#     def __init__(
#         self,
#         root: str,
#         subset: str,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ):
#         super(CIFAR10P, self).__init__(root, transform, target_transform, download)
#
#         self.subset = subset
#
#         if subset not in self.subsets and subset != "all":
#             raise ValueError(f"Unknown Subset: {subset}")
#
#         if subset == "all":
#             self.data = np.concatenate(
#                 [np.load(join(root, self.base_folder, f"{s}.npy")) for s in self.subsets]
#             )
#         else:
#             self.data = np.load(join(root, self.base_folder, f"{subset}.npy"))
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img = self.data[index]
#         target = -1
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target


class CIFAR100C(ImageDatasetBase):
    """
    From the paper *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.*

    :see Website: https://zenodo.org/record/3555552
    """

    base_folder = "CIFAR-100-C/"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
    tgz_md5 = "11f0ed0f1191edbf9fa23466ae6021d3 "
