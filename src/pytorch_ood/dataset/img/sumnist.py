import logging
import os.path
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url

log = logging.getLogger(__name__)


class SuMNIST(Dataset):
    """
    The SuMNIST dataset comprises images with a size of :math:`56 \\times 56`, each containing 4 numbers from the MNIST
    dataset. In the training dataset, there are 60,000 normal instances where the numbers in the image sum to 20.
    However, the test set with 10,000 images, there are 8,500 normal instances and 1,500 anomalous
    instances for which the numbers do not sum to 20. The challenge is to detect these anomalies.

    Returns a tuple with ``(img, dict)``  where dict contains bounding boxes, labels, etc.


    :see Paper: `LNCS <https://link.springer.com/chapter/10.1007/978-3-031-40953-0_32>`__
    :see Examples: `GitHub <https://github.com/kkirchheim/sumnist>`__

    .. image:: https://github.com/kkirchheim/sumnist/blob/master/img/mnist-example.png?raw=true
        :width: 800px
        :alt: SuMNIST Dataset examples
        :align: center

    """

    url = "https://files.kondas.de/sumnist/"

    files = {
        "b-test.npz": "85a544301eff979e252b8946e31fd795",
        "b-train.npz": "44b6208a8675df1a78c981b5ad8c4e50",
        "x-test.npz": "7db6727ec075cca1bb4dd0881087ac57",
        "x-train.npz": "c087f1c74a6f7ffcad9956be6f99cf10",
        "y-test.npz": "7239555b3d809657c06fbbc8da6f3e5e",
        "y-train.npz": "c6c94eb5ed7ebbe1c466abb14712e807",
    }

    base_dir = "sumnist"

    def __init__(self, root, train=True, transforms=None, download=False):
        """

        :param root: where to store dataset
        :param train: set to `False` to use test set
        :param transforms: callable to apply to image and target dictionary
        :param download: set to `True` to download automatically
        """
        self.root = join(root, SuMNIST.base_dir)
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        if train:
            with np.load(join(self.root, "x-train.npz")) as data:
                self.x = torch.tensor(data["arr_0"])

            with np.load(join(self.root, "y-train.npz")) as data:
                self.y = torch.tensor(data["arr_0"])

            with np.load(join(self.root, "b-train.npz")) as data:
                self.b = torch.tensor(data["arr_0"])
        else:
            with np.load(join(self.root, "x-test.npz")) as data:
                self.x = torch.tensor(data["arr_0"])

            with np.load(join(self.root, "y-test.npz")) as data:
                self.y = torch.tensor(data["arr_0"])

            with np.load(join(self.root, "b-test.npz")) as data:
                self.b = torch.tensor(data["arr_0"])

    def __len__(self):
        return len(self.x)

    def _check_integrity(self) -> bool:
        for file, hash in SuMNIST.files.items():
            fpath = os.path.join(self.root, file)
            if not check_integrity(fpath, hash):
                return False

        return True

    def download(self):
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        for file, hash in SuMNIST.files.items():
            url = SuMNIST.url + file
            download_url(url, self.root, md5=hash)

    def __getitem__(self, index):
        img = self.x[index]
        img = img.repeat(3, 1, 1)  # To RGB

        bboxes = self.b[index]

        boxes = []
        for box in bboxes:
            x_min, x_max, y_min, y_max = box
            boxes.append((x_min, y_min, x_max, y_max))

        bboxes = boxes

        labels = self.y[index]

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = labels = torch.zeros((len(boxes),), dtype=torch.int64)
        target["anomaly"] = torch.tensor(-1 if labels.sum().item() != 20 else 0).long()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
