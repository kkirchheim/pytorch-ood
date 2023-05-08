import logging
import os
from os.path import join
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class ImageNetA(DatasetFolder):
    """
    From the paper *Natural Adversarial Examples*.
    Contains images that classifiers should be able to classify

    :see Website: `GitHub <https://github.com/hendrycks/natural-adv-examples>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1907.07174>`__
    """

    base_folder = "imagenet-a"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    filename = "imagenet-a.tar"
    tgz_md5 = "c3e55429088dc681f30d81f4726b6595"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        loader = Image.open
        super(ImageNetA, self).__init__(
            root=join(root, self.base_folder),
            loader=loader,
            is_valid_file=lambda x: x.endswith(".jpg") or x.endswith(".JPEG"),
            transform=transform,
            target_transform=target_transform,
        )

    def _check_integrity(self) -> bool:
        return check_integrity(join(self.root, self.filename), self.tgz_md5)

    def download(self) -> None:
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class ImageNetO(ImageNetA):
    """
    From the paper *Natural Adversarial Examples*.
    Contains anomalies of unforeseen classes

    :see Website: `GitHub <https://github.com/hendrycks/natural-adv-examples>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1907.07174>`__
    """

    base_folder = "imagenet-o"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
    filename = "imagenet-o.tar"
    tgz_md5 = "86bd7a50c1c4074fb18fc5f219d6d50b"


class ImageNetR(ImageNetA):
    """
    The ImageNet-R(endition) from the paper *The Many Faces of Robustness: A Critical
    Analysis of Out-of-Distribution Generalization* contains art, cartoons, deviantart,
    graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects,
    plush objects, sculptures, sketches, tattoos, toys, and video game renditions of ImageNet classes.


    :see Website: `GitHub <https://github.com/hendrycks/imagenet-r>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/2006.16241>`__
    """

    base_folder = "imagenet-r"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    filename = "imagenet-r.tar"
    tgz_md5 = "a61312130a589d0ca1a8fca1f2bd3337"


class ImageNetC(ImageDatasetBase):
    """
    Corrupted version of the ImageNet from the paper *Benchmarking Neural
    Network Robustness to Common Corruptions and Perturbations.*

    It contains several subsets:

    * ``noise`` (21GB): gaussian_noise, shot_noise, and impulse_noise.
    * ``blur`` (7GB): defocus_blur, glass_blur, motion_blur, and zoom_blur.
    * ``weather`` (12GB):  frost, snow, fog, and brightness.
    * ``digital`` (7GB): contrast, elastic_transform, pixelate, and jpeg_compression.
    * ``extra`` (15GB): speckle_noise, spatter, gaussian_blur, and saturate.

    :see Paper: `ArXiv <https://arxiv.org/abs/1903.12261v1>`__
    """

    subset_list = ["blur", "digital", "extra", "noise", "weather"]

    base_folder_list = [
        "ImageNetC/blur/",
        "ImageNetC/digital/",
        "ImageNetC/extra/",
        "ImageNetC/noise/",
        "ImageNetC/weather/",
    ]
    url_list = [
        "https://zenodo.org/record/2235448/files/blur.tar",
        "https://zenodo.org/record/2235448/files/digital.tar",
        "https://zenodo.org/record/2235448/files/extra.tar",
        "https://zenodo.org/record/2235448/files/noise.tar",
        "https://zenodo.org/record/2235448/files/weather.tar",
    ]
    filename_list = ["blur.tar", "digital.tar", "extra.tar", "noise.tar", "weather.tar"]
    tgz_md5_list = [
        "2d8e81fdd8e07fef67b9334fa635e45c",
        "89157860d7b10d5797849337ca2e5c03",
        "d492dfba5fc162d8ec2c3cd8ee672984",
        "e80562d7f6c3f8834afb1ecf27252745",
        "33ffea4db4d93fe4a428c40a6ce0c25d",
    ]

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        if subset not in self.subset_list:
            raise ValueError(f"Invalid subset: {subset}")

        self.base_folder = self.base_folder_list[self.subset_list.index(subset)]
        self.url = self.url_list[self.subset_list.index(subset)]
        self.filename = self.filename_list[self.subset_list.index(subset)]
        self.tgz_md5 = self.tgz_md5_list[self.subset_list.index(subset)]

        super(ImageDatasetBase, self).__init__(
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
