import logging
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import ImageNet, VisionDataset

from .base import _get_resource_file

log = logging.getLogger(__name__)


def _load_classes() -> List[str]:
    """Return the 800 ImageNet-800 WNIDs, in sorted order."""
    with open(_get_resource_file("imagenet800_classes.txt")) as f:
        return [line.strip() for line in f if line.strip()]


class ImageNet800(VisionDataset):
    """
    ImageNet-800 comprises the 800 ImageNet-1K classes that are *not* part of
    :class:`ImageNet200`, i.e. the two class sets are disjoint and together cover all of
    ImageNet-1K. Samples are intended to
    be used as auxiliary/outlier data during training (e.g. with
    :class:`pytorch_ood.loss.OutlierExposureLoss`), not as a standalone classification task,
    so ``__getitem__`` always returns a target of ``-1``.

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        :param root: root of a torchvision-compatible ImageNet directory (with ``train/``,
            ``val/`` and the devkit), as used by :class:`torchvision.datasets.ImageNet`
        :param transform: transform applied to images
        :param target_transform: transform applied to targets
        """
        super(ImageNet800, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.classes = _load_classes()  #: 800 WNIDs, disjoint from :class:`ImageNet200`
        self.samples: List[str] = self._make_samples()

    def _make_samples(self) -> List[str]:
        wnids = set(self.classes)
        base = ImageNet(self.root, split="train")
        return [path for path, idx in base.samples if base.wnids[idx] in wnids]

    def __getitem__(self, index: int) -> Tuple[object, int]:
        path = self.samples[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        target = -1
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.samples)
