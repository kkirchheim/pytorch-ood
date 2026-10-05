"""
Some of the datasets used in OpenOOD 1.5 benchmark.

"""

import json
import logging
import os
from os.path import exists, join
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets.utils import extract_archive

from pytorch_ood.api import DatasetInfo, Paper, Role, Task
from pytorch_ood.dataset.img.base import ImageDatasetBase, _get_resource_file

log = logging.getLogger(__name__)


class OpenOOD(ImageDatasetBase):
    """
    Abstract Base Class for OpenOOD datasets. The data is downloaded from Google Drive, which requires ``gdown``.
    """

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
            (raises a :class:`RuntimeError` if ``gdown`` is not installed)
        """
        self.archive_file = join(root, self.filename)

        super(OpenOOD, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def download(self) -> None:
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        try:
            import gdown

            gdown.download(id=self.gdrive_id, output=self.archive_file)
        except ImportError:
            raise RuntimeError("You have to install 'gdown' to download this dataset")

        extract_archive(from_path=self.archive_file, to_path=join(self.root, self.target_dir))

    def _check_integrity(self) -> bool:
        return exists(self.archive_file)


class iNaturalist(OpenOOD):
    """
    Subset of the iNaturalist dataset used as OOD data for ImageNet, proposed in
    *MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space*.

    Images are returned as :class:`PIL.Image.Image`. All labels are -1 by default.

    :see Paper: `The iNaturalist Species Classification and Detection Dataset <https://openaccess.thecvf.com/content_cvpr_2018/html/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.html>`__
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license="iNaturalist 2017 competition terms (non-commercial research and education)",
        paper=Paper(
            title="MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space",
            venue="CVPR",
            year=2021,
            url="https://arxiv.org/abs/2105.01879",
        ),
    )

    gdrive_id = "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj"
    filename = "iNaturalist.zip"
    target_dir = "iNaturalist"
    base_folder = join(target_dir, "images")

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
        super(iNaturalist, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class OpenImagesO(OpenOOD):
    """
    Images sourced from the OpenImages dataset used as OOD data for ImageNet, as provided in
    *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection*.
    All labels are -1 by default.

    The test set contains 15,869 images, the validation set 1,763 images.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license="Annotations CC-BY-4.0, images CC-BY-2.0 (verify individually)",
        paper=Paper(
            title="OpenOOD: Benchmarking Generalized Out-of-Distribution Detection",
            venue="NeurIPS",
            year=2022,
            url="https://arxiv.org/abs/2210.07242",
        ),
        homepage="https://storage.googleapis.com/openimages/web/index.html",
    )

    gdrive_id = "1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE"
    filename = "openimage_o.zip"
    target_dir = "OpenImagesO"
    base_folder = join(target_dir, "images")

    inclusion_json = {
        "test": "test_openimage_o.json",
        "val": "val_openimage_o.json",
    }

    def __init__(
        self,
        root: str,
        subset="test",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param subset: can be either ``val`` or ``test``
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target
        :param download: download the data to ``root`` if it is not found there
        :raises AssertionError: if ``subset`` is invalid
        """
        assert subset in list(self.inclusion_json.keys())
        super(OpenImagesO, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        p = _get_resource_file(self.inclusion_json[subset])
        with open(p, "r") as f:
            included = json.load(f)

        self.files = [join(self.basedir, f) for f in included]


class Places365(OpenOOD):
    """
    Images sourced from the Places365 dataset used as OOD data, usually for CIFAR 10 and 100.
    All labels are -1 by default.

    The dataset contains 36,500 images.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license="Places2 terms (non-commercial research and education)",
        homepage="http://places.csail.mit.edu/browser.html",
    )

    gdrive_id = "1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay"
    filename = "places365.zip"
    target_dir = "places365"
    base_folder = target_dir

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
        super(Places365, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.files = []

        for d in sorted(os.listdir(self.basedir)):
            p = join(self.basedir, d)
            if not os.path.isdir(join(p)):
                continue
            self.files += [join(p, f) for f in sorted(os.listdir(p))]


class ImageNetV2(OpenOOD):
    """
    A new test set for ImageNet, introduced in  *Do ImageNet Classifiers Generalize to ImageNet?*.
    While it contains no OOD data, it is utilized for evaluating OOD detection methods.

    The test set consists of 10000 images across 1000 classes, with 10 images per class.
    Images are returned as :class:`PIL.Image.Image`. Targets are the ImageNet class indices, not ``-1``.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license=None,
        paper=Paper(
            title="Do ImageNet Classifiers Generalize to ImageNet?",
            venue="ICML",
            year=2019,
            url="https://arxiv.org/abs/1902.10811",
        ),
    )

    gdrive_id = "1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho"
    filename = "imagenet_v2.zip"
    target_dir = "imagenet_v2"
    base_folder = target_dir

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param download: download the data to ``root`` if it is not found there
        """
        super(ImageNetV2, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.basedir = join(root, self.base_folder)

        # iterate over folders in the base folder
        self.files = []
        self.labels = []
        for class_folder in sorted(os.listdir(self.basedir)):
            # folder name is the class id
            class_folder_path = join(self.basedir, class_folder)
            # skip if not a folder
            if not os.path.isdir(class_folder_path):
                continue
            # add all images in the folder to files
            for img in sorted(os.listdir(class_folder_path)):
                self.files.append(join(class_folder_path, img))
                self.labels.append(int(class_folder))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = self.labels[index]
        return img, label


class ImageNetES(OpenOOD):
    """
    ImageNet under covariate shifts in the environment (lighting) and in the camera's sensor settings (ES),
    introduced in *Unexplored Faces of Robustness and Out-of-Distribution: Covariate Shifts in Environment
    and Sensor Domains*.
    While it contains no OOD data, it is utilized for evaluating OOD detection methods.

    The provided data here is similar to that in the OpenOOD benchmark, making it only a subset of the original dataset.

    The test set consists of 64000 images across 200 different classes.
    Images are returned as :class:`PIL.Image.Image`. Targets are class indices, not ``-1``.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.DISTRIBUTION_SHIFT},
        license=None,
        paper=Paper(
            title="Unexplored Faces of Robustness and Out-of-Distribution: Covariate Shifts in Environment and Sensor Domains",
            venue="CVPR",
            year=2024,
            url="https://arxiv.org/abs/2404.15882",
            code="https://github.com/Edw2n/ImageNet-ES",
        ),
    )

    gdrive_id = "1ATz11vKmPqyzfEaEDRaPTF9TXiC244sw"
    filename = "imagenet_es.zip"
    target_dir = "imagenet_es"
    base_folder = join(target_dir, "es-test")
    data_file = "imagenet_es.json"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param download: download the data to ``root`` if it is not found there
        """
        super(ImageNetES, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.basedir = join(root, self.base_folder)

        self.files, self.labels = self.load_and_check_images()

    def load_and_check_images(self):
        p = _get_resource_file(self.data_file)
        # read json file
        with open(p, "r") as file:
            data = json.load(file)

        # iterate over the dictionary and check if the images exist
        images = []
        labels = []
        basedir = join(self.root, self.base_folder)
        for folder1 in sorted(os.listdir(basedir)):
            # skip folder "sampled_tin_no_resize2"
            if folder1 == "sampled_tin_no_resize2":
                continue
            for folder2 in sorted(os.listdir(join(basedir, folder1))):
                for folder3 in sorted(os.listdir(join(basedir, folder1, folder2))):
                    for class_tag in sorted(os.listdir(join(basedir, folder1, folder2, folder3))):
                        for img in sorted(
                            os.listdir(join(basedir, folder1, folder2, folder3, class_tag))
                        ):
                            images.append(join(basedir, folder1, folder2, folder3, class_tag, img))
                            labels.append(data[class_tag])

        return images, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = self.labels[index]
        return img, label


class SSBHard(OpenOOD):
    """
    The SSB-hard is the hard split of the Semantic Shift Benchmark (SSB), introduced in *Open-set recognition: A good closed-set classifier is all you need*.
    This dataset only provides OOD data and is used for open-set recognition for models trained on ImageNet1K.

    The test set consists of 49000 images. Images are returned as :class:`PIL.Image.Image`;
    all labels are -1 by default.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license=None,
        paper=Paper(
            title="Open-Set Recognition: a Good Closed-Set Classifier is All You Need?",
            venue="ICLR",
            year=2022,
            url="https://arxiv.org/abs/2110.06207",
        ),
    )

    gdrive_id = "1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE"
    filename = "ssb_hard.zip"
    target_dir = "ssb_hard"
    base_folder = target_dir

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
        super(SSBHard, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.basedir = join(root, self.base_folder)

        self.files = []
        for class_folder in sorted(os.listdir(self.basedir)):
            # folder name is the class id
            class_folder_path = join(self.basedir, class_folder)
            # skip if not a folder
            if not os.path.isdir(class_folder_path):
                continue
            # add all images in the folder to files
            for img in sorted(os.listdir(class_folder_path)):
                self.files.append(join(class_folder_path, img))
