import logging
import os
from os.path import join
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from ...api import DatasetInfo, Paper, Role, Task
from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class ImageNetA(DatasetFolder):
    """
    From the paper *Natural Adversarial Examples*.
    Contains 7,500 natural adversarial images of 200 ImageNet classes that a ResNet-50 misclassifies.

    Images are returned as :class:`PIL.Image.Image`. Targets are the class indices of the folder
    structure (sorted folder names), not ``-1``.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license=None,
        paper=Paper(
            title="Natural Adversarial Examples",
            venue="CVPR",
            year=2021,
            url="https://arxiv.org/abs/1907.07174",
        ),
        homepage="https://github.com/hendrycks/natural-adv-examples",
    )

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
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target (the class index)
        :param download: download the data to ``root`` if it is not found there
        """
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
    Contains 2,000 images of classes that are not among the 200 ImageNet classes of ImageNet-A, and thus can be
    used as OOD data for models trained on ImageNet.

    Targets are the class indices of the folder structure, **not** ``-1``. When using this dataset as OOD data,
    mark the samples as OOD with ``target_transform=ToUnknown()`` (see :class:`pytorch_ood.utils.ToUnknown`).
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license=None,
        paper=Paper(
            title="Natural Adversarial Examples",
            venue="CVPR",
            year=2021,
            url="https://arxiv.org/abs/1907.07174",
        ),
        homepage="https://github.com/hendrycks/natural-adv-examples",
    )

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

    Images are returned as :class:`PIL.Image.Image`. Targets are the class indices of the folder
    structure (sorted folder names), not ``-1``.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license=None,
        paper=Paper(
            title="The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization",
            venue="ICCV",
            year=2021,
            url="https://arxiv.org/abs/2006.16241",
        ),
        homepage="https://github.com/hendrycks/imagenet-r",
    )

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

    Each subset has to be downloaded and loaded separately.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license="CC-BY-4.0",
        paper=Paper(
            title="Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
            venue="ICLR",
            year=2019,
            url="https://arxiv.org/abs/1903.12261",
        ),
    )

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
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param subset: one of ``blur``, ``digital``, ``extra``, ``noise`` and ``weather``
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target
        :param download: download the data to ``root`` if it is not found there
        :raises ValueError: if ``subset`` is invalid
        """
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
        self.files = sorted(os.listdir(self.basedir))
