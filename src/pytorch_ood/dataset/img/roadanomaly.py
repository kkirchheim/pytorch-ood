import logging
import os
from os.path import join
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.transforms.functional import to_tensor

from ...api import DatasetInfo, Paper, Role, Task
from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class RoadAnomaly(ImageDatasetBase):
    """
    Benchmark Dataset for Anomaly Segmentation.

    From the paper *Detecting the Unexpected via Image Resynthesis*. Contains 60 images.

    Images are :class:`PIL.Image.Image` of size :math:`1280 \\times 720`. The target is a float tensor of shape
    :math:`H \\times W` with value ``0`` for in-distribution pixels and ``-1`` for anomalous pixels.

    .. image:: https://www.epfl.ch/labs/cvlab/wp-content/uploads/2019/10/road_anomaly_gt_contour-1024x576.jpg
        :width: 800px
        :alt: RoadAnomaly Dataset Example
        :align: center
    """

    info = DatasetInfo(
        task=Task.SEGMENTATION,
        roles={Role.BENCHMARK},
        license="Research use only (images are not owned by the authors)",
        paper=Paper(
            title="Detecting the Unexpected via Image Resynthesis",
            venue="ICCV",
            year=2019,
            url="https://arxiv.org/abs/1904.07595",
        ),
        homepage="https://www.epfl.ch/labs/cvlab/data/road-anomaly/",
    )

    root_dir_name = "RoadAnomaly"

    url = "https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly_jpg.zip"

    filename = ("RoadAnomaly_jpg.zip", "87a0908e5c72827824693913cf2e4fb0")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: directory in which the data is stored, or looked up if it was downloaded before
        :param transform: called as ``transform(image, mask)`` with the PIL image and the target mask,
            and must return the transformed ``(image, mask)`` tuple
        :param download: download the data to ``root`` if it is not found there
        """
        root = join(root, self.root_dir_name)
        super(ImageDatasetBase, self).__init__(root, transform=transform)

        self.filename, self.tgz_md5 = self.filename

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )
        self.data_dir = join(root, "RoadAnomaly_jpg")

        self.all_images, self.all_masks = self._get_file_list(self.data_dir)

    def __len__(self) -> int:
        return len(self.all_images)

    def _get_file_list(self, root) -> List[str]:
        """
        Recursively get all files in the root directory

        :param root: root directory for the search
        """
        current_files = [entry for entry in os.listdir(join(root, "frames"))]

        all_images = []
        all_masks = []

        for path in current_files:
            if path.endswith(".jpg"):
                all_images.append(join(root, "frames", path))
                all_masks.append(
                    join(
                        root,
                        "frames",
                        f"{path.replace('.jpg', '')}.labels",
                        "labels_semantic.png",
                    )
                )

        assert len(all_images) == len(all_masks)
        if len(all_images) == 0:
            log.error("No images found in the directory")
        if len(all_masks) == 0:
            log.error("No masks found in the directory")
        if len(all_images) != len(all_masks):
            raise Exception(
                f"Number of images and masks do not match: num_img:{len(all_images)}, num_masks:{len(all_masks)}"
            )
        if len(all_images) != 60:
            raise Exception(f"Not Enough Images are found: {len(all_images)}")
        return all_images, all_masks

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: index
        :return: tuple of the image and the target mask of shape :math:`H \\times W`, ``-1`` marks anomalies
        """
        file, target = self.all_images[index], self.all_masks[index]

        # to return a PIL Image
        img = Image.open(file)
        target = to_tensor(Image.open(target)).squeeze(0)
        # all values above 0 are outliers
        target[target > 0] = -1  # negative labels for outliers

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
