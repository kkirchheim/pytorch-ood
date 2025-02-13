"""

"""

import logging
from os.path import exists, join

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

log = logging.getLogger(__name__)


class CIFAR100GAN(Dataset):
    """
    Images sampled from low likelihood regions of a BigGAN trained on CIFAR 100 from the paper
    *On Outlier Exposure with Generative Models*.

    Can be used as auxiliary outliers, e.g. for :class:`OutlierExposure <pytorch_ood.loss.OutlierExposureLoss>` or
    any of the supervised training objectives in general.

    Default sample :math:`\\sigma` is 50.0. Contains 50,000 samples. Label is `-1` by default.


    .. image :: https://files.kondas.de/goe-data/cifar100gan.jpg
        :width: 600px
        :alt: CIFAR 100 GAN Dataset
        :align: center

    :see Website: `GitHub <https://github.com/kkirchheim/mlsw2022-goe>`__
    :see Paper: `NeurIPS MLSW <https://openreview.net/forum?id=SU7OAfhc8OM>`__

    """

    filename = {2.0: "samples-2.0.npz", 50.0: "samples-50.0.npz"}

    url = {
        2.0: "https://files.kondas.de/goe-data/samples-2.0.npz",
        50.0: "https://files.kondas.de/goe-data/samples-50.0.npz",
    }

    md5 = {
        2.0: "f130876edbbc13ab2bdc6f7caaa1180d",
        50.0: "95f1365e4c6e188595bb8476d43a82d9",
    }

    def __init__(self, root, transform=None, target_transform=None, download=False, sigma=50.0):
        """
        :param root: where to store the dataset
        :param transform: transform to apply to the data
        :param target_transform: transform to apply to the target
        :param download: whether to download the dataset if it is not found in root
        :param sigma: sample :math:`\\sigma` used to generate dataset. Can be ``50.0`` or ``2.0``.
        """
        self.datafile = join(root, self.filename[sigma])
        if not exists(self.datafile):
            if download:
                download_url(self.url[sigma], root, md5=self.md5[sigma])
            else:
                raise FileNotFoundError("Missing File. Set download=True to download.")

        log.debug(f"Loading data from {self.datafile}")
        self.data = np.load(self.datafile)["x"]
        log.debug(f"Shape of dataset: {self.data.shape}")
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        index = index % len(self)
        img = self.data[index]
        label = -1

        img = np.moveaxis(img, 0, -1)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return self.data.shape[0]
