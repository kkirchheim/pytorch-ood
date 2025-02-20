"""

..  autoclass:: pytorch_ood.utils.ToUnknown
    :members:

..  autoclass:: pytorch_ood.utils.ToRGB
    :members:

..  autoclass:: pytorch_ood.utils.TargetMapping
    :members:

"""

from typing import Set, Callable, Union

import os
import random
from os.path import join

import numpy as np
from PIL import Image, ImageDraw
import torch
from collections import defaultdict
import json
from typing import List, Tuple

from torch import Tensor
from torchvision.datasets.utils import download_and_extract_archive


class ToUnknown(object):
    """
    Callable that returns a negative number, used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1


class ToRGB(object):
    """
    Convert Image to RGB, if it is not already.
    """

    def __call__(self, x):
        try:
            return x.convert("RGB")
        except Exception as e:
            return x


class TargetMapping(object):
    """
    Maps ID (a.k.a. known) classes to labels :math:`\\in [0,n]`, and OOD (a.k.a. unknown) classes to labels in :math:`[-\\infty, -1]`.
    This is required for open set simulations.

    **Example:**
    If we split up a dataset so that the classes 2,3,4,9 are considered *known* or *ID*, these class
    labels have to be remapped to 0,1,2,3 to be able to train
    using cross entropy with 1-of-K-vectors. All other classes have to be mapped to values :math:`<0`
    to be marked as OOD.
    """

    def __init__(self, known: Set, unknown: Set):
        self._map = dict()
        self._map.update({clazz: index for index, clazz in enumerate(set(known))})
        # mapping train_out classes to < 0
        self._map.update({clazz: (-clazz) for index, clazz in enumerate(set(unknown))})

    def __call__(self, target):
        if isinstance(target, torch.Tensor):
            return self._map.get(target.item(), -1)

        return self._map.get(target, -1)

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            return self._map[item.item()]

        return self._map[item]

    def items(self):
        return self._map.items()

    def __repr__(self):
        return str(self._map)
