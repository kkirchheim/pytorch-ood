import logging
import os
from os.path import exists, join

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from ...api import DatasetInfo, Role, Task

log = logging.getLogger(__name__)


class TinyImageNet(VisionDataset):
    """
    Small Version of the ImageNet with images of size :math:`64 \\times 64` from 200 classes used by
    Stanford. Each class has 500 images for training.

    This dataset is often used for training, but not included in Torchvision.
    Images are returned as :class:`PIL.Image.Image`. The ``train`` and ``val`` subsets return the class index
    :math:`0, \\dots, 199` as target; the ``test`` subset has no annotations, so all its targets are ``-1``
    (the label of OOD samples).
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.IN_DISTRIBUTION, Role.OOD_TEST},
        license=None,
        homepage="http://cs231n.stanford.edu/",
    )

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dir_name = "tiny-imagenet-200"
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    filename = "tiny-imagenet-200.zip"
    subsets = ["train", "val", "test"]

    def __init__(
        self,
        root,
        subset="train",
        download=False,
        transform=None,
        target_transform=None,
    ):
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param subset: can be one of ``train``, ``val`` and ``test``
        :param download: download the data to ``root`` if it is not found there
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target
        :raises ValueError: if ``subset`` is invalid
        """
        if subset not in self.subsets:
            raise ValueError(f"Invalid subset: {subset}. Possible values are {self.subsets}")

        super(TinyImageNet, self).__init__(
            root, target_transform=target_transform, transform=transform
        )

        self.subset = subset

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        classes = os.listdir(join(self.root, self.dir_name, "train"))
        classes.sort()
        self.class_map = {c: n for n, c in enumerate(classes)}  # : map class_names to integers
        self.basename = join(self.root, self.dir_name, self.subset)
        self.paths = []
        self.labels = []

        if subset == "train":
            for d in classes:
                p = join(self.basename, d, "images")
                files = [join(p, img) for img in os.listdir(p)]

                self.paths += files
                self.labels += [self.class_map[d]] * len(files)

        elif subset == "val":
            anno_file = join(self.basename, "val_annotations.txt")
            with open(anno_file, "r") as f:
                for line in f.readlines():
                    path, label, x, y, z, t = " ".join(line.split()).split()
                    self.paths.append(join(self.basename, "images", path))
                    self.labels.append(self.class_map[label])

        elif subset == "test":
            d = join(self.basename, "images")
            self.paths = [join(d, img) for img in os.listdir(d)]
            self.labels = [-1] * len(self.paths)

    def download(self):
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _check_integrity(self):
        return exists(join(self.root, self.dir_name))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the class index in :math:`[0, 199]` for the ``train`` and ``val``
            subsets and ``-1`` for the unlabeled ``test`` subset.
        """
        img, target = self.paths[index], self.labels[index]

        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.paths)
