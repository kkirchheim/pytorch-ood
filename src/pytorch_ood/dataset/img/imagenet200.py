import logging
from os.path import basename
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import ImageNet, VisionDataset

from .base import _get_resource_file

log = logging.getLogger(__name__)


def _load_classes() -> List[str]:
    """Return the 200 ImageNet-200 WNIDs in OpenOOD label order (sorted)."""
    with open(_get_resource_file("imagenet200_classes.txt")) as f:
        return [line.strip() for line in f if line.strip()]


class ImageNet200(VisionDataset):
    """
    The ImageNet-200 in-distribution dataset used by the OpenOOD v1.5 benchmark.

    ImageNet-200 is the 200-class subset of ImageNet-1K whose classes are identical to
    those of ImageNet-R. Class labels are assigned ``0..199`` in sorted WNID order, matching
    the labelling used by OpenOOD (and the classifiers it provides).

    This dataset is a *view* on a standard, torchvision-compatible ImageNet directory: it
    reuses :class:`torchvision.datasets.ImageNet` to locate images and then filters/relabels
    them. The original ImageNet (with devkit) must already be present at ``root``; the data is
    not downloaded.

    The OpenOOD splits are reproduced exactly:

     * ``train`` -- all ImageNet-train images of the 200 classes (~259k images)
     * ``val`` -- 1000 held-out ImageNet-val images (5 per class) for hyperparameter tuning
     * ``test`` -- 9000 ImageNet-val images (45 per class), the in-distribution test set

    :see Paper: `OpenOOD v1.5 <https://arxiv.org/abs/2306.09301>`__
    """

    splits = ("train", "val", "test")

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        :param root: root of a torchvision-compatible ImageNet directory (with ``train/``,
            ``val/`` and the devkit), as used by :class:`torchvision.datasets.ImageNet`
        :param split: one of ``train``, ``val`` or ``test``
        :param transform: transform applied to images
        :param target_transform: transform applied to targets
        """
        super(ImageNet200, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {self.splits}")

        self.split = split
        self.classes = _load_classes()  #: 200 WNIDs, index == label
        self.wnid_to_label = {wnid: idx for idx, wnid in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = self._make_samples()

    def _make_samples(self) -> List[Tuple[str, int]]:
        if self.split == "train":
            base = ImageNet(self.root, split="train")
            samples = []
            for path, idx in base.samples:
                label = self.wnid_to_label.get(base.wnids[idx])
                if label is not None:
                    samples.append((path, label))
            return samples

        # val / test: a fixed subset of the ImageNet validation images. The image basename
        # is unique across the val split; OpenOOD provides the basename -> label mapping.
        resource = "imagenet200_val.txt" if self.split == "val" else "imagenet200_test.txt"
        base = ImageNet(self.root, split="val")
        path_by_name = {basename(path): path for path, _ in base.samples}

        samples = []
        with open(_get_resource_file(resource)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, label = line.rsplit(" ", 1)
                samples.append((path_by_name[name], int(label)))
        return samples

    def __getitem__(self, index: int) -> Tuple[object, int]:
        path, target = self.samples[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.samples)
