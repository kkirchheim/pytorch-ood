import logging

import numpy as np
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class TinyImages(Dataset):
    """
    The TinyImages dataset is often used as auxiliary OOD training data.
    While it has been removed from the website, downloadable versions can be found on the internet.

    :see Website: https://groups.csail.mit.edu/vision/TinyImages/
    :see Archive: https://archive.org/details/80-million-tiny-images-1-of-2

    ..  warning::
        The use of this dataset is discouraged by the authors.
        See *Large image datasets: A pyrrhic win for computer vision?*

    """

    def __init__(self, datafile, cifar_index_file, transform=None, exclude_cifar=True):
        self.datafile = datafile
        self.cifar_index_file = cifar_index_file
        self.n_images = 79302017  #

        data_file = open(self.datafile, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype="uint8").reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0  # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(self.cifar_index_file, "r") as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % self.n_images

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(self.n_images)

        while True:
            try:
                img = self.load_image(index)
                if self.transform is not None:
                    img = self.transform(img)

                return img, -1  # 0 is the class
            except ValueError as e:
                # log.warning(f"Failed to read image {index}")
                index = np.random.randint(self.n_images)
            except Exception as e:
                log.warning(f"Failed to read image {index}")
                log.exception(e)
                index = np.random.randint(self.n_images)

                if self.exclude_cifar:
                    while self.in_cifar(index):
                        index = np.random.randint(self.n_images)

    def __len__(self):
        return self.n_images


class TinyImages300k(Dataset):
    """
    A cleaned version of the TinyImages Dataset

    :see Page: https://github.com/hendrycks/outlier-exposure
    """

    def __init__(self, datafile, transform=None):
        self.datafile = datafile

        log.info(f"Loading data from {datafile}")
        self.data = np.load(datafile)
        log.info(f"Shape of dataset: {self.data.shape}")
        self.transform = transform

    def __getitem__(self, index):
        index = index % len(self)

        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, -1

    def __len__(self):
        return self.data.shape[0]
