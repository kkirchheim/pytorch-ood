#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Dataset
"""
import numpy as np
import torch.utils.data as data
from PIL import Image


class UniformNoise(data.Dataset):
    """Dataset that outputs gaussian noise only"""

    def __init__(self, num, size=(224, 224, 3), transform=None, target_transform=None, **kwargs):
        self.size = size
        self.num = num
        self.transform = transform
        self.target_transform = target_transform
        self.target = -1

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = np.random.uniform(low=0, high=255, size=self.size).astype(dtype=np.uint8)

        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        target = self.target

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
           target = self.target_transform(target)

        return img, target