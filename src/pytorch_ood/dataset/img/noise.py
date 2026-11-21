#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Image Datasets
"""

from abc import ABC
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ...api import DatasetInfo, Role, Task


class NoiseDataset(Dataset, ABC):
    """
    Base Class for noise datasets. Sample :math:`i` is generated from the pair (seed, :math:`i`)
    alone, so it is the same however often, in which order and in which worker process it is
    drawn.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        :param seed: non-negative integer that determines all samples. With ``None``, a seed is
            drawn from OS entropy once at construction and stored in ``seed``.
        """
        # OS entropy, not the global RNG; mirrors what numpy does for seed=None
        self.seed: int = np.random.SeedSequence().entropy if seed is None else seed
        self.default_target = -1

    def _generator(self, index: int) -> np.random.Generator:
        """
        A fresh generator for sample ``index``, so that samples do not share generator state.
        """
        n = len(self)
        if not -n <= index < n:
            raise IndexError(f"index {index} out of range for dataset of length {n}")
        # SeedSequence hashes the pair, so neighbouring indices or seeds give unrelated streams
        return np.random.default_rng([self.seed, index % n])


class GaussianNoise(NoiseDataset):
    """
    Dataset with images drawn from a normal distribution. Images are returned as :class:`PIL.Image.Image` with
    values clipped to :math:`[0, 255]`; all targets are ``-1`` (the label of OOD samples) by default.
    The same index always gives the same image, also across epochs and :class:`DataLoader <torch.utils.data.DataLoader>` workers.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license=None,
    )

    def __init__(
        self,
        length: int,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        loc: int = 128,
        scale: int = 128,
        seed: Optional[int] = None,
    ):
        """
        :param length: number of samples in the dataset
        :param size: shape :math:`H \\times W \\times C` of each generated image; with :math:`C=1`, images are grayscale
        :param transform: transformation to apply to images
        :param target_transform: transformation to apply to labels
        :param loc: mean :math:`\\mu` of the gaussian
        :param scale: standard deviation :math:`\\sigma` of the gaussian
        :param seed: non-negative integer; sample :math:`i` depends only on the seed and
            :math:`i`. With ``None``, a seed is drawn once at construction and stored in ``seed``.
        """
        super(GaussianNoise, self).__init__(seed=seed)

        if not isinstance(length, int):
            raise ValueError("length parameter must be an integer")

        self.size = size
        self.num = length
        self.transform = transform
        self.target_transform = target_transform
        self.loc = loc
        self.scale = scale

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = self._generator(item).normal(loc=self.loc, scale=self.scale, size=self.size)
        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
        img = np.clip(img, 0, 255).astype("uint8")
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        target = self.default_target
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class UniformNoise(NoiseDataset):
    """
    Dataset with images drawn from a uniform distribution over :math:`[0, 255]`. Images are returned as
    :class:`PIL.Image.Image`; all targets are ``-1`` (the label of OOD samples) by default.
    The same index always gives the same image, also across epochs and :class:`DataLoader <torch.utils.data.DataLoader>` workers.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license=None,
    )

    def __init__(
        self,
        length: int,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        seed: Optional[int] = None,
    ):
        """
        :param length: number of samples in the dataset
        :param size: shape :math:`H \\times W \\times C` of each generated image; with :math:`C=1`, images are grayscale
        :param transform: transformation to apply to images
        :param target_transform: transformation to apply to labels
        :param seed: non-negative integer; sample :math:`i` depends only on the seed and
            :math:`i`. With ``None``, a seed is drawn once at construction and stored in ``seed``.
        """

        super(UniformNoise, self).__init__(seed=seed)

        if not isinstance(length, int):
            raise ValueError("length parameter must be an integer")

        self.size = size
        self.num = length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = self._generator(item).uniform(low=0, high=255, size=self.size).astype(dtype=np.uint8)
        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        target = self.default_target
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
