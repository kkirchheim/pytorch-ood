#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Dataset
"""
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image


class GaussianNoise(VisionDataset):
    """Dataset that outputs gaussian noise only"""

    def __init__(
        self, num, size=(224, 224, 3), transform=None, target_transform=None, **kwargs
    ):
        print(f"Noise Dataset with Size: {size}")
        self.size = size
        self.num = num
        self.transform = transform
        self.target_transform = target_transform
        self.target = -1

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = np.random.normal(loc=128, scale=128, size=self.size)

        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))

        img = np.clip(img, 0, 255).astype("uint8")

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        target = self.target
        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #    target = self.target_transform(target)

        return img, target
