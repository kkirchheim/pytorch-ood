""" """

import os
from os.path import join
from typing import Callable, Optional

from ...api import DatasetInfo, Paper, Role, Task
from .base import ImageDatasetBase


class FoolingImages(ImageDatasetBase):
    """
    From the paper *Deep neural networks are easily fooled: High confidence predictions for unrecognizable images*.

    .. image::  https://i.stack.imgur.com/pBm48.png
        :width: 800px
        :alt: Fooling Images
        :align: center
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license=None,
        paper=Paper(
            title="Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images",
            venue="CVPR",
            year=2015,
            url="https://arxiv.org/abs/1412.1897",
        ),
        homepage="https://anhnguyen.me/project/fooling/",
    )

    dirs = [f"run_{i}" for i in range(10)]

    base_folder = "10-runs-x-1000-cppns"
    url = "https://s.anhnguyen.me/10_runs_x_1000_cppns.tar.gz"
    filename = "10_runs_x_1000_cppns.tar.gz"
    md5hash = "0910d63973b1512770f37bebdbb53e37"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(FoolingImages, self).__init__(root, transform, target_transform, download)

    def _load_files(self):
        self.basedir = os.path.join(self.root, self.base_folder)
        files = []
        for d in self.dirs:
            p = join(self.basedir, d, "map_gen_5000")
            files += [join(p, f) for f in os.listdir(p) if f.endswith(".png")]

        return files
