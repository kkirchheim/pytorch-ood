import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from ...api import DatasetInfo, Paper, Role, Task

log = logging.getLogger(__name__)


class Textures(VisionDataset):
    """
    Textures dataset from the paper *Describing Textures in the Wild*, also known as DTD.
    Often used as OOD data. Images are returned as :class:`PIL.Image.Image`; all targets are ``-1``
    (the label of OOD samples) by default, use ``target_transform`` to change them.
    """

    info = DatasetInfo(
        task=Task.CLASSIFICATION,
        roles={Role.OOD_TEST},
        license=None,
        paper=Paper(
            title="Describing Textures in the Wild",
            venue="CVPR",
            year=2014,
            url="https://arxiv.org/abs/1311.3618",
        ),
        homepage="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
    )

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
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param transform: function applied to the image (a :class:`PIL.Image.Image`)
        :param target_transform: function applied to the target
        :param download: download the data to ``root`` if it is not found there
        """
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
        for d in sorted(os.listdir(self.basedir)):
            self.files.extend(
                [
                    join(d, f)
                    for f in sorted(os.listdir(join(self.basedir, d)))
                    if not f.startswith(".")
                ]
            )
        log.info(f"Found {len(self.files)} texture files.")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is ``-1`` (the label of OOD samples).
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
