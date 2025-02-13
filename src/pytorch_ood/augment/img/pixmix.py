"""
Much of this code is directly taken or adapted from https://github.com/andyzoujm/pixmix/
which is licensed under MIT according to the gitlab repo, however, some of the files have an
apache 2.0 license header. Both should be compatible with our license.
"""

import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive
from torchvision.transforms.functional import to_tensor

log = logging.getLogger(__name__)


# NOTE: list of augmentations used originally
# augmentations_all = [
#     F.autocontrast, F.equalize, F.posterize, F.rotate, F.solarize, F.shear_x, F.shear_y,
#     F.translate_x, F.translate_y, F.color, F.contrast, F.brightness, F.sharpness
# ]


def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1**a) * (img2.clip(1e-37) ** b)
    return out / 2


class PixMixDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper to perform PixMix, from the paper
    *PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures*.

    :see Paper: `ArXiv <https://arxiv.org/abs/2112.05135>`__

    .. note ::
        Some of the augmentations primitives used in the paper are not yet implemented.

    .. image:: https://github.com/andyzoujm/pixmix/raw/main/assets/pixmix.png
        :width: 800px
        :alt: Pixmix sketch
        :align: center

    """

    def __init__(
        self,
        dataset,
        mixing_set,
        beta=3,
        aug_severity=3,
        k=4,
        std=(1.0, 1.0, 1.0),
        mean=(1.0, 1.0, 1.0),
    ):
        """
        :param dataset: original dataset
        :param mixing_set: dataset used for mixing
        :param beta: mixing coefficient
        :param aug_severity: severity used for augmentation primitives
        :param k: number of mixing iterations
        :param mean: used for normalization
        :param std: used for normalization
        """
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.normalize = torchvision.transforms.Normalize(std=std, mean=mean)
        self.mixings = [add, multiply]
        self.augmentations = [
            lambda x, y: F.equalize(x),
            lambda x, y: F.autocontrast(x),
            F.posterize,
            F.rotate,
            F.solarize,
        ]
        self.beta = beta
        self.aug_severity = aug_severity
        self.k = k

    def __getitem__(self, i):
        x, y = self.dataset[i]
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        return self._pixmix(x, mixing_pic), y

    def __len__(self):
        return len(self.dataset)

    def _pixmix(self, orig, mixing_pic):
        """
        :param orig: original image
        :param mixing_pic: picture to mix in
        """
        # first, apply one of the augmentations with 50% chance to the original image
        # TODO: make probability configurable?
        if np.random.random() < 0.5:
            mixed = to_tensor(self.augment_input(orig, severity=self.aug_severity))
        else:
            mixed = to_tensor(orig)

        # then k times: create an augmented copy of the original, use use the mixing pic
        # mix whatever you generated into the original image
        for _ in range(np.random.randint(self.k + 1)):
            if np.random.random() < 0.5:
                aug_image_copy = to_tensor(self.augment_input(orig, severity=self.aug_severity))
            else:
                aug_image_copy = to_tensor(mixing_pic)

            # mix current image and augmented copy
            mixed_op = np.random.choice(self.mixings)
            mixed = mixed_op(mixed, aug_image_copy, self.beta)
            mixed = torch.clip(mixed, 0, 1)

        return self.normalize(mixed)

    def augment_input(self, image, severity):
        op = np.random.choice(self.augmentations)
        return op(image.copy(), severity)
