import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class Textures(VisionDataset):
    """
    Textures dataset from the paper *Describing Textures in the Wild*, also known as DTD.
    Often used as OOD data.

    .. image :: https://production-media.paperswithcode.com/datasets/DTD-0000002377-abe5e400_AubcN36.jpg
        :width: 600px
        :alt: Textured Dataset
        :align: center

    :see Paper: `ArXiv <https://arxiv.org/abs/1311.3618v2>`__
    :see Website: `Link <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`__
    """

    base_folder = "dtd/images/"
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    filename = "textures-r1_0_1.tar.gz"
    tgz_md5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(Textures, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.basedir = join(self.root, self.base_folder)
        self.files = []
        for d in os.listdir(self.basedir):
            self.files.extend(
                [join(d, f) for f in os.listdir(join(self.basedir, d)) if not f.startswith(".")]
            )
        log.info(f"Found {len(self.files)} texture files.")

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
        path = join(self.root, self.base_folder, file)
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

        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
