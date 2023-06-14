#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Image Datasets
"""
from abc import ABC

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class NoiseDataset(Dataset, ABC):
    """
    Base Class for noise datasets
    """

    def __init__(self, seed: int = None):
        """
        :param seed: seed used to initialize the random number generator
        """
        self.generator = np.random.default_rng(seed=seed)
        self.default_target = -1


class GaussianNoise(NoiseDataset):
    """
    Dataset with samples drawn from a normal distribution.
    """

    def __init__(
        self,
        length: int,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        loc: int = 128,
        scale: int = 128,
        seed: int = None,
    ):
        """
        :param length: number of samples in the dataset
        :param size: shape of the generated noise samples
        :param transform: transformation to apply to images
        :param target_transform: transformation to apply to labels
        :param loc: mean :math:`\\mu` of the gaussian
        :param scale: scaling factor :math:`\\sigma^2` of the gaussian
        :param seed: random seed
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
        img = self.generator.normal(loc=self.loc, scale=self.scale, size=self.size)
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
    Dataset with samples drawn from uniform distribution.
    """

    def __init__(
        self,
        length: int,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        seed: int = None,
    ):
        """
        :param length: number of samples in the dataset
        :param size: shape of the generated noise samples
        :param transform: transformation to apply to images
        :param target_transform: transformation to apply to labels
        :param seed: random seed
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
        img = self.generator.uniform(low=0, high=255, size=self.size).astype(dtype=np.uint8)
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
