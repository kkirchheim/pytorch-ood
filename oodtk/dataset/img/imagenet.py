import logging
import os
import sys
from typing import Optional, Callable, Any, Tuple

sys.path.insert(0, os.path.join(os.getcwd(),"oodtk", "dataset", "img"))
from base import ImageDatasetHandler

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)



class ImageNetA(ImageDatasetHandler):
    """
    Natural Adversarial Examples

    :see Website: https://github.com/hendrycks/natural-adv-examples

    # https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
    """
    base_folder = "ImageNetA/images/"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    filename = "imagenet-a.tar"
    tgz_md5 = "c3e55429088dc681f30d81f4726b6595"

class ImageNetO(ImageDatasetHandler):
    """
    Natural Adversarial Examples for OOD

    :see Website: https://github.com/hendrycks/natural-adv-examples

    # https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar
    """
    base_folder = "ImageNetO/images/"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
    filename = "imagenet-o.tar"
    tgz_md5 = "86bd7a50c1c4074fb18fc5f219d6d50b"

class ImageNetR(ImageDatasetHandler):
    """

    :see Website: https://github.com/hendrycks/imagenet-r

    https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar

    The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization

    :see Paper: https://arxiv.org/abs/2006.16241

    """
    base_folder = "ImageNetR/images/"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    filename = "imagenet-r.tar"
    tgz_md5 = "a61312130a589d0ca1a8fca1f2bd3337"

class ImageNetC(ImageDatasetHandler):
    """
    BENCHMARKING NEURAL NETWORK ROBUSTNESS TO COMMON CORRUPTIONS AND PERTURBATIONS

    noise.tar (21GB) contains gaussian_noise, shot_noise, and impulse_noise.
    blur.tar (7GB) contains defocus_blur, glass_blur, motion_blur, and zoom_blur.
    weather.tar (12GB) contains frost, snow, fog, and brightness.
    digital.tar (7GB) contains contrast, elastic_transform, pixelate, and jpeg_compression.
    extra.tar (15GB) contains speckle_noise, spatter, gaussian_blur, and saturate.


    https://zenodo.org/record/2235448/files/blur.tar?download=1
    md5:2d8e81fdd8e07fef67b9334fa635e45c 

    https://zenodo.org/record/2235448/files/digital.tar?download=1
    md5:89157860d7b10d5797849337ca2e5c03 

    https://zenodo.org/record/2235448/files/extra.tar?download=1
    md5:d492dfba5fc162d8ec2c3cd8ee672984

    https://zenodo.org/record/2235448/files/noise.tar?download=1
    md5:e80562d7f6c3f8834afb1ecf27252745

    https://zenodo.org/record/2235448/files/weather.tar?download=1
    md5:33ffea4db4d93fe4a428c40a6ce0c25d 

    """
    subset_list = ['blur', 'digital', 'extra', 'noise', 'weather']

    base_folder_list = ["ImageNetC/blur/","ImageNetC/digital/", "ImageNetC/extra/", "ImageNetC/noise/", "ImageNetC/weather/" ]
    url_list = [
        "https://zenodo.org/record/2235448/files/blur.tar", 
        "https://zenodo.org/record/2235448/files/digital.tar", 
        "https://zenodo.org/record/2235448/files/extra.tar",
        "https://zenodo.org/record/2235448/files/noise.tar",
        "https://zenodo.org/record/2235448/files/weather.tar"
        ]
    filename_list = [
        "blur.tar",
        "digital.tar",
        "extra.tar", 
        "noise.tar",
        "weather.tar"
        ]
    tgz_md5_list = [
        "2d8e81fdd8e07fef67b9334fa635e45c",
        "89157860d7b10d5797849337ca2e5c03",
        "d492dfba5fc162d8ec2c3cd8ee672984",
        "e80562d7f6c3f8834afb1ecf27252745",
        "33ffea4db4d93fe4a428c40a6ce0c25d"
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        subset: str=None,) -> None:
        
        self.base_folder = self.base_folder_list[self.subset_list.index(subset)]
        self.url = self.url_list[self.subset_list.index(subset)]
        self.filename = self.filename_list[self.subset_list.index(subset)]
        self.tgz_md5 = self.tgz_md5_list[self.subset_list.index(subset)]

        super(ImageDatasetHandler, self).__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = os.listdir(self.basedir)

class ImageNetP(ImageDatasetHandler):
    """
    BENCHMARKING NEURAL NETWORK ROBUSTNESS TO COMMON CORRUPTIONS AND PERTURBATIONS
    """
    subset_list = ['blur', 'digital', 'noise', 'weather']

    base_folder_list = ["ImageNetP/blur/", "ImageNetP/digital/", "ImageNetP/noise/", "ImageNetP/weather/" ]
    url_list = [
        "https://zenodo.org/record/3565846/files/blur.tar", 
        "https://zenodo.org/record/3565846/files/digital.tar", 
        "https://zenodo.org/record/3565846/files/noise.tar",
        "https://zenodo.org/record/3565846/files/weather.tar"
        ]
    filename_list = [
        "blur.tar",
        "digital.tar",
        "noise.tar",
        "weather.tar"
        ]
    tgz_md5_list = [
        "869b2d2fb1604d1baa4316b5ecc9fdea ",
        "8c1f73e4912812788e2fdb637cd55372 ",
        "619b37b5139ce764f77ce3bed3aee837",
        "a597ff6db8c6fb0dbcd6eb12ed620002"
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        subset: str=None,) -> None:
        
        self.base_folder = self.base_folder_list[self.subset_list.index(subset)]
        self.url = self.url_list[self.subset_list.index(subset)]
        self.filename = self.filename_list[self.subset_list.index(subset)]
        self.tgz_md5 = self.tgz_md5_list[self.subset_list.index(subset)]

        super(ImageDatasetHandler, self).__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = os.listdir(self.basedir)
