"""
Some of the datasets used in OpenOOD 1.5 benchmark.

"""

import json
import logging
import os
from os.path import dirname, exists, join
from typing import Callable, Optional
from PIL import Image

from torchvision.datasets.utils import extract_archive

from pytorch_ood.dataset.img.base import ImageDatasetBase, _get_resource_file

log = logging.getLogger(__name__)


class OpenOOD(ImageDatasetBase):
    """
    Abstract Base Class for OpenOOD datasets.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
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

    All labels are -1 by default.

    :see Paper: `MOS <https://arxiv.org/pdf/2105.01879.pdf>`__
    :see Paper: `iNaturalist <https://openaccess.thecvf.com/content_cvpr_2018/html/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.html>`__


    """

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

    :see Website: `OpenImages <https://storage.googleapis.com/openimages/web/index.html>`__

    The test set contains 15869 , the validation set 1763 images.
    """

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
        :param subset: can be either ``val`` or ``test``
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

    Dataset set contains 36500 images.

    :see Website: `Places <http://places.csail.mit.edu/browser.html>`__

    .. image:: https://production-media.paperswithcode.com/datasets/Places-0000003475-4b6da14b.jpg
      :target: http://places.csail.mit.edu/browser.html
      :alt: Places 365 examples

    """

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
        super(Places365, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.files = []

        for d in os.listdir(self.basedir):
            p = join(self.basedir, d)
            if not os.path.isdir(join(p)):
                continue
            self.files += [join(p, f) for f in os.listdir(p)]


class ImageNetV2(OpenOOD):
    """
    A new test set for ImageNet, introduced in  *Do ImageNet Classifiers Generalize to ImageNet?*.
    While it contains no OOD data, it is utilized for evaluating OOD detection methods.

    :see Paper: `ArXiv <https://arxiv.org/pdf/1902.10811>`__


    The test set consists of 10000 images across 1000 classes, with 10 images per class.
    """

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
        for class_folder in os.listdir(self.basedir):
            # folder name is the class id
            class_folder_path = join(self.basedir, class_folder)
            # skip if not a folder
            if not os.path.isdir(class_folder_path):
                continue
            # add all images in the folder to files
            for img in os.listdir(class_folder_path):
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
    A new test set for ImageNet as event-stream (ES) version, introduced in *ES-ImageNet: A Million Event-Stream
    Classification Dataset for Spiking Neural Networks*.
    While it contains no OOD data, it is utilized for evaluating OOD detection methods.


    The provided data here is similar to that in the OpenOOD benchmark, making it only a subset of the original dataset.

    :see Paper: `ArXiv <https://arxiv.org/pdf/2110.12211>`__


    The test set consists of 64000 images across 200 different classes.
    """

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
        for folder1 in os.listdir(basedir):
            # skip folder "sampled_tin_no_resize2"
            if folder1 == "sampled_tin_no_resize2":
                continue
            for folder2 in os.listdir(join(basedir, folder1)):
                for folder3 in os.listdir(join(basedir, folder1, folder2)):
                    for class_tag in os.listdir(join(basedir, folder1, folder2, folder3)):
                        for img in os.listdir(join(basedir, folder1, folder2, folder3, class_tag)):
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

    :see Paper: `ArXiv <https://arxiv.org/pdf/2110.06207>`__


    The test set consists of 49000 images.
    """

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
        super(SSBHard, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.basedir = join(root, self.base_folder)

        self.files = []
        for class_folder in os.listdir(self.basedir):
            # folder name is the class id
            class_folder_path = join(self.basedir, class_folder)
            # skip if not a folder
            if not os.path.isdir(class_folder_path):
                continue
            # add all images in the folder to files
            for img in os.listdir(class_folder_path):
                self.files.append(join(class_folder_path, img))
