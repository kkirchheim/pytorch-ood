import os
from os.path import join
from typing import Optional, Callable, Tuple, Any

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive

import logging


log = logging.getLogger(__name__)


class PixMixExampleDatasets(VisionDataset):
    google_drive_id = "1qC2gIUx9ARU7zhgI4IwGD3YcFhm8J4cA"
    filename = "fractals_and_fvis.tar"
    subdirs = {
        "fractals": "fractals/images/",
        "features": "first_layers_resized256_onevis/images/",
    }
    base_folder = "fractals_and_fvis"
    md5sum = "3619fb7e2c76130749d97913fdd3ab27"

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(PixMixExampleDatasets, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if subset not in self.subdirs.keys():
            raise ValueError(f"Invalid subset '{subset}'. Allowed: {list(self.subdirs.keys())}")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.basedir = join(self.root, self.base_folder, self.subdirs[subset])
        self.files = os.listdir(self.basedir)

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
        path = os.path.join(self.basedir, file)
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.files)

    def _check_integrity(self) -> bool:
        fpath = os.path.join(self.root, self.filename)
        return check_integrity(fpath, self.md5sum)

    def download(self) -> None:
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        try:
            import gdown

            archive = os.path.join(self.root, self.filename)
            if not gdown.download(id=self.google_drive_id, output=archive):
                raise Exception("File must be downloaded manually")

            log.info("Extracting {archive} to {self.root}")
            extract_archive(archive, self.root, remove_finished=False)

        except ImportError:
            raise ImportError("You have to install 'gdown' to use this dataset.")


class FeatureVisDataset(PixMixExampleDatasets):
    """
    Dataset with Feature visualizations, as used in
    *PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures*.

    :see Paper: `ArXiv <https://arxiv.org/abs/2112.05135>`__
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download=False,
    ) -> None:
        super(FeatureVisDataset, self).__init__(
            root,
            subset="features",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )


class FractalDataset(PixMixExampleDatasets):
    """
    Dataset with Fractals, as used in
    *PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures*.

    :see Paper: `ArXiv <https://arxiv.org/abs/2112.05135>`__
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download=False,
    ) -> None:
        super(FractalDataset, self).__init__(
            root,
            subset="fractals",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
