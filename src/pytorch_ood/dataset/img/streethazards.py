import logging
import os
from os.path import join
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.transforms.functional import to_tensor

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class StreetHazards(ImageDatasetBase):
    """
    Benchmark Dataset for Anomaly Segmentation.

    From the paper *Scaling Out-of-Distribution Detection for Real-World Settings*

    .. image:: https://github.com/hendrycks/anomaly-seg/raw/master/streethazards.gif
        :width: 800px
        :alt: Street Hazards Dataset Example
        :align: center

    :see Paper: `ArXiv <https://arxiv.org/.*>`__
    :see Website: `GitHub <https://github.com/hendrycks/anomaly-seg>`__
    """

    classes: List[str] = [
        "unlabeled",
        "building",
        "fence",
        "other",
        "pedestrian",
        "pole",
        "road line",
        "road",
        "sidewalk",
        "vegetation",
        "car",
        "wall",
        "traffic sign",
    ]  #: class index to name mapping

    subset_list = ["test", "train", "validation"]

    root_dir_name = "streethazards"

    base_folders = {
        "test": "test/images/",
        "train": "train/images/training/",
        "validation": "train/images/validation/",
    }

    url_list = {
        "test": "https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar",
        "train": "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar",
        "validation": "https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar",
    }

    filename_list = {
        "test": ("streethazards_test.tar", "8c547c1346b00c21b2483887110bfea7"),
        "train": ("streethazards_train.tar", "cd2d1a8649848afb85b5059d227d2090"),
        "validation": ("streethazards_train.tar", "cd2d1a8649848afb85b5059d227d2090"),
    }

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable[[Tuple], Tuple]] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root path for dataset
        :param subset: one of ``train``, ``test``, ``validation``
        :param transform: transformations to apply to images and masks, will get tuple as argument
        :param download: if dataset should be downloaded automatically
        """
        root = join(root, self.root_dir_name)
        super(ImageDatasetBase, self).__init__(root, transform=transform)

        self.base_folder = self.base_folders[subset]
        self.url = self.url_list[subset]
        self.filename, self.tgz_md5 = self.filename_list[subset]

        if download:
            self.download()

        if subset not in self.subset_list:
            raise ValueError(f"Invalid subset: {subset}")

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.basedir = os.path.join(self.root, self.base_folder)

        self.files = self._get_file_list(self.basedir)

    def _get_file_list(self, root) -> List[str]:
        """
        Recursively get all files in the root directory

        :param root: root directory for the search
        """
        current_files = [os.path.join(root, entry) for entry in os.listdir(root)]
        all_files = []

        for path in current_files:
            if os.path.isdir(path):
                all_files += self._get_file_list(path)
            else:
                all_files.append(path)

        return all_files

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: index
        :returns: (image, target) where target is the annotation of the image.
        """
        file, target = self.files[index], self.files[index].replace("images", "annotations")

        # to return a PIL Image
        img = Image.open(file)
        target = to_tensor(Image.open(target)).squeeze(0)
        target = (target * 255).long() - 1  # labels to integer
        target[target >= 13] = -1  # negative labels for outliers

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
