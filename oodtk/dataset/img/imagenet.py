import logging
import os
from typing import Optional, Callable, Any, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class ImageDatasetHandler(VisionDataset):
    """
    Base Class for Downlaoding ImageNet related Datasets
    
    Code inspired from : https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10

    """
    base_folder = ""
    url = ""
    filename = ""
    tgz_md5 = ""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(ImageDatasetHandler, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = os.listdir(self.basedir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file, target = self.files[index], -1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        path = os.path.join(self.root, self.base_folder, file)
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.files)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, self.tgz_md5)

    def download(self) -> None:
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

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

    Create a hashmap to download subset images and directory

    """

    base_folder = ["ImageNetC/blur/","ImageNetC/digital/", "ImageNetC/extra/", "ImageNetC/noise/", "ImageNetC/weather/" ]
    url = [
        "https://zenodo.org/record/2235448/files/blur.tar", 
        "https://zenodo.org/record/2235448/files/digital.tar", 
        "https://zenodo.org/record/2235448/files/extra.tar",
        "https://zenodo.org/record/2235448/files/noise.tar",
        "https://zenodo.org/record/2235448/files/weather.tar"
        ]
    filename = [
        "blur.tar",
        "digital.tar",
        "extra.tar", 
        "noise.tar",
        "weather.tar"
        ]
    tgz_md5 = [
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
    ) -> None:
        super(ImageDatasetHandler, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)
        self.files = os.listdir(self.basedir)


class ImageNetP:
    """
    BENCHMARKING NEURAL NETWORK ROBUSTNESS TO COMMON CORRUPTIONS AND PERTURBATIONS
    """
