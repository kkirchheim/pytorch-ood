import logging
import os
from os.path import join
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.transforms.functional import to_tensor

from .base import ImageDatasetBase

log = logging.getLogger(__name__)


class SegmentMeIfYouCan(ImageDatasetBase):
    """
    Benchmark Dataset for Anomaly Segmentation.

    From the paper *SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation*. Contains two subsets: RoadAnomaly21 and RoadObstacle21

    .. note:: Similar to Paper *Segment Every Out-of-Distribution Object* (`ArXiv <https://arxiv.org/pdf/2311.16516v3>`__, `Github <https://github.com/WenjieZhao1/S2M>`__) for ``RoadAnomaly21`` only **10** and for ``RoadObstacle21`` only **30** images are available.


    :see Paper: `ArXiv <https://arxiv.org/pdf/2104.14812>`__
    :see Website: `Website <https://segmentmeifyoucan.com/datasets>`__
    """

    root_dir_name = "SMIYC"
    subset_list = ["RoadAnomaly21", "RoadObstacle21"]

    base_folders = {
        "RoadAnomaly21": "dataset_AnomalyTrack",
        "RoadObstacle21": "dataset_ObstacleTrack",
    }

    dataset_length = {"RoadAnomaly21": 10, "RoadObstacle21": 30}

    url_list = {
        "RoadAnomaly21": "https://zenodo.org/record/5270237/files/dataset_AnomalyTrack.zip",
        "RoadObstacle21": "https://zenodo.org/record/5281633/files/dataset_ObstacleTrack.zip",
    }

    filename_list = {
        "RoadAnomaly21": (
            "dataset_AnomalyTrack.zip",
            "231bf79ed58924bcd33d9cbe22e61076",
        ),
        "RoadObstacle21": (
            "dataset_ObstacleTrack.zip",
            "895fb36d18765482cc291f69e63d6da6",
        ),
    }
    VOID_LABEL = 1  #: void label, should be ignored during score calculation

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable[[Tuple], Tuple]] = None,
        download: bool = False,
    ) -> None:
        """
        :param root: root path for dataset
        :param subset: one of ``RoadAnomaly21``, ``RoadObstacle21``
        :param transform: transformations to apply to images and masks, will get tuple as argument
        :param download: if dataset should be downloaded automatically
        """
        root = join(root, self.root_dir_name)
        super(ImageDatasetBase, self).__init__(root, transform=transform)

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
        self.data_dir = join(root, self.base_folders[subset])

        self.all_images, self.all_masks = self._get_file_list(self.data_dir, subset)

        assert self.dataset_length[subset] == len(self.all_images)

    def __len__(self) -> int:
        return len(self.all_images)

    def _get_file_list(self, root, subset) -> List[str]:
        """
        Recursively get all files in the root directory

        :param root: root directory for the search
        """
        current_files = [entry for entry in os.listdir(join(root, "labels_masks"))]

        all_images = []
        all_masks = []
        if subset == "RoadAnomaly21":
            for path in current_files:
                if path.endswith(".png") and "color" not in path:
                    all_images.append(join(root, "images", path.split("_")[0] + ".jpg"))
                    all_masks.append(join(root, "labels_masks", path))
        if subset == "RoadObstacle21":
            for path in current_files:
                if path.endswith(".png") and "color" not in path:
                    all_images.append(
                        join(
                            root,
                            "images",
                            path.split("_")[0] + "_" + path.split("_")[1] + ".webp",
                        )
                    )
                    all_masks.append(join(root, "labels_masks", path))
        assert len(all_images) == len(all_masks)
        if len(all_images) == 0:
            log.error("No images found in the directory")
        if len(all_masks) == 0:
            log.error("No masks found in the directory")
        if len(all_images) != len(all_masks):
            raise Exception(
                f"Number of images and masks do not match: num_img:{len(all_images)}, num_masks:{len(all_masks)}"
            )

        return all_images, all_masks

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: index
        :returns: (image, target) where target is the annotation of the image.
        """
        file, target = self.all_images[index], self.all_masks[index]

        # to return a PIL Image
        img = Image.open(file)
        target = to_tensor(Image.open(target)).squeeze(0)
        target = (target * 255).long()
        # void pixels to -10
        target[target == 255] = -10  # -10 labels for ignore
        # all values above 0 are outliers
        target[target > 0] = -1  # negative labels for outliers
        # set void labels
        target[target == -10] = self.VOID_LABEL  # void labels for ignore

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
