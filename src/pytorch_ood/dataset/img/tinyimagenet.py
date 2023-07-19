import logging
import os
from os.path import exists, join
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


log = logging.getLogger(__name__)


class TinyImageNet(VisionDataset):
    """
    Small Version of the ImageNet with images of size :math:`64 \\times 64` from 200 classes used by
    Stanford. Each class has 500 images for training.

    .. image :: https://production-media.paperswithcode.com/datasets/Tiny_ImageNet-0000001404-a53923c3_XCrVSGm.jpg
        :width: 400px
        :alt: Textured Dataset
        :align: center


    This dataset is often used for training, but not included in Torchvision.

    :see Website: `Stanford <http://cs231n.stanford.edu/>`__

    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dir_name = "tiny-imagenet-200"
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    filename = "tiny-imagenet-200.zip"
    subsets = ["train", "val", "test"]

    def __init__(self, root, subset="train", download=False, transform=None, target_transform=None):
        """
        :para subset: can be one of ``train``, ``val`` and ``test``
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
            tuple: (image, target) where target is index of the target class.
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

