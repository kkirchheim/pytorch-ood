import logging
import os
from typing import Any, Callable, Optional, Tuple

from base import ImageDatasetHandler
from torchvision.datasets import VisionDataset

log = logging.getLogger(__name__)


class MNISTC(VisionDataset):
    """
    MNIST-C is MNIST with corruptions for benchmarking OOD methods.

    :see Paper: https://arxiv.org/pdf/1906.02337.pdf
    :see Download: https://zenodo.org/record/3239543
    """

    # url = "https://zenodo.org/record/3239543/files/mnist_c.zip?download=1"

    subset_list = ["test", "leftovers"]

    base_folder_list = ["MNISTC/images/", "MNISTC_leftovers/images/"]
    url_list = [
        "https://zenodo.org/record/3239543/files/mnist_c.zip",
        "https://zenodo.org/record/3239543/files/mnist_c_leftovers.zip",
    ]
    filename_list = [
        "mnist_c.zip",
        "mnist_c_leftovers.zip",
    ]
    tgz_md5_list = [
        "4b34b33045869ee6d424616cd3a65da3 ",
        "c365e9c25addd5c24454b19ac7101070 ",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        subset: str = None,
    ) -> None:

        self.base_folder = self.base_folder_list[self.subset_list.index(subset)]
        self.url = self.url_list[self.subset_list.index(subset)]
        self.filename = self.filename_list[self.subset_list.index(subset)]
        self.tgz_md5 = self.tgz_md5_list[self.subset_list.index(subset)]

        super(ImageDatasetHandler, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = os.listdir(self.basedir)
