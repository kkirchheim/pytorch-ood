import logging
import os
from os.path import join
from typing import Callable, Optional

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


    :see Paper: `ICML <https://arxiv.org/pdf/2306.00826.pdf>`__
    :see Code: `GitHub <https://github.com/j-cb/NINCO>`__
    :see Download: `Zenodo <https://zenodo.org/record/8013288>`__

    """

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
            for subdir in os.listdir(path):
                files += [join(path, subdir, img) for img in os.listdir(join(path, subdir))]

        return files
