#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Dataset
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class GaussianNoise(Dataset):
    """
    Dataset that outputs gaussian noise
    """

    def __init__(
        self,
        length,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        loc=128,
        scale=128,
    ):
        self.size = size
        self.num = length
        self.transform = transform
        self.target_transform = target_transform
        self.loc = loc
        self.scale = scale

    def __len__(self):
        return self.num

    def __getitem__(self, item) -> Image:
        img = np.random.normal(loc=self.loc, scale=self.scale, size=self.size)
        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
        img = np.clip(img, 0, 255).astype("uint8")
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        target = 0
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class UniformNoise(Dataset):
    """
    Dataset that outputs uniform noise
    """

    def __init__(
        self, length, size=(224, 224, 3), transform=None, target_transform=None, **kwargs
    ):
        self.size = size
        self.num = length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num

    def __getitem__(self, item) -> Image:
        img = np.random.uniform(low=0, high=255, size=self.size).astype(dtype=np.uint8)
        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        target = 0
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
