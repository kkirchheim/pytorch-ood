"""
Dataset that loads images listed in an OpenOOD-style imglist file.
"""

import logging
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset

log = logging.getLogger(__name__)


def _default_loader(path: str) -> Image.Image:
    return Image.open(path)


class ImageListDataset(VisionDataset):
    """
    Dataset that loads images listed in a plain-text image list file.

    Each non-empty, non-comment line of ``imglist_path`` has the form
    ``<relative_path> <label>``, where ``<relative_path>`` is resolved
    against ``root``. This format is used by OpenOOD and OpenMIBOOD.

    For non-PIL formats (e.g. NIfTI ``.nii.gz`` volumes used by OASIS-3),
    pass a custom ``loader`` callable.

    .. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
       :alt: slop-badge
    """

    def __init__(
        self,
        root: str,
        imglist_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """
        :param root: directory the relative paths in ``imglist_path`` are resolved against
        :param imglist_path: path to a text file with ``<relative_path> <label>`` per line
        :param transform: transform applied to each loaded image
        :param target_transform: transform applied to each label
        :param loader: callable mapping a file path to an image; defaults to :func:`PIL.Image.open`
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if not os.path.isdir(root):
            raise RuntimeError(f"Data root directory not found: {root}")
        if not os.path.isfile(imglist_path):
            raise RuntimeError(f"Image list file not found: {imglist_path}")

        self.loader = loader or _default_loader

        self.files = []
        self.labels = []
        with open(imglist_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                self.files.append(os.path.join(root, parts[0]))
                self.labels.append(int(parts[1]))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(self.files[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
