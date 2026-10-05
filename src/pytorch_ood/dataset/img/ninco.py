import logging
import os
from os.path import join
from typing import Callable, Optional

from pytorch_ood.api import DatasetInfo, Paper, Role, Task
from pytorch_ood.dataset.img.base import ImageDatasetBase

log = logging.getLogger(__name__)


class NINCO(ImageDatasetBase):
    """
    NINCO dataset from the paper
    *In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation*. Contains 5879 OOD images from 64 classes.
    The images have been manually verified as OOD.

    Labels are -1 by default.

    .. note :: Calculating metrics over the entire dataset will result in slightly different results compared
        to the original publication, as they calculate metrics over each class individually and
        report the mean.

    :see Download: `Zenodo <https://zenodo.org/record/8013288>`__
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license="CC-BY-4.0",
        paper=Paper(
            title="In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation",
            venue="ICML",
            year=2023,
            url="https://arxiv.org/abs/2306.00826",
            code="https://github.com/j-cb/NINCO",
        ),
    )

    base_folders = [
        "NINCO_OOD_classes"
    ]  # , "NINCO_OOD_unit_tests", "NINCO_popular_datasets_subsamples"
    url = "https://zenodo.org/record/8013288/files/NINCO_all.tar.gz"
    filename = "NINCO_all.tar.gz"
    tgz_md5s = "b9ffae324363cd900a81ce3c367cd834"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target
        :param download: download the data to ``root`` if it is not found there
        """
        super(NINCO, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def _load_files(self):
        files = []
        for d in self.base_folders:
            path = join(self.root, "NINCO", d)
            for subdir in sorted(os.listdir(path)):
                files += [
                    join(path, subdir, img) for img in sorted(os.listdir(join(path, subdir)))
                ]

        return files
