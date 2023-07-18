"""
Some of the datasets used in OpenOOD 1.5 benchmark.

"""
import json
import os
from typing import Callable, Optional

from os.path import exists, join, dirname
from torchvision.datasets.utils import extract_archive
import logging


from pytorch_ood.dataset.img.base import ImageDatasetBase, _get_resource_file

log = logging.getLogger(__name__)


class iNaturalist(ImageDatasetBase):
    """
    Subset of the iNaturalist dataset used as OOD data for ImageNet, proposed in
    *MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space*.

    All labels are -1 by default.

    :see Paper: `MOS <https://arxiv.org/pdf/2105.01879.pdf>`__
    :see Paper: `iNaturalist <https://openaccess.thecvf.com/content_cvpr_2018/html/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.html>`__


    """
    gdrive_id = "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj"
    filename = "iNaturalist.zip"
    target_dir = "iNaturalist"
    base_folder = join(target_dir, "images")

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:

        self.archive_file = join(root, self.filename)
        super(iNaturalist, self).__init__(root=root, transform=transform, target_transform=target_transform,
                                          download=download)

    def download(self) -> None:
        if self._check_integrity():
            log.debug("Files already downloaded and verified")
            return

        try:
            import gdown
            gdown.download(id=self.gdrive_id, output=self.archive_file)
        except ImportError:
            raise RuntimeError("You have to install 'gdown' to download this dataset")

        extract_archive(from_path=self.archive_file, to_path=join(self.root, self.target_dir))

    def _check_integrity(self) -> bool:
        return exists(join(self.root, self.filename))


class OpenImagesO(iNaturalist):
    """
    Images sourced from the OpenImages dataset used as OOD data for ImageNet, as provided in
    *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection*.
    All labels are -1 by default.

    :see Website: `OpenImages <https://storage.googleapis.com/openimages/web/index.html>`__

    The test set contains 15869 , the validation set 1763 images.
    """
    gdrive_id = "1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE"
    filename = "openimage_o.zip"
    target_dir = "OpenImagesO"
    base_folder = join(target_dir, "images")

    inclusion_json = {
        "test": "test_openimage_o.json",
        "val": "val_openimage_o.json",
    }

    def __init__(self,
                 root: str,
                 subset="test",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:
        """
        :param subset: can be either ``val`` or ``test``
        """
        assert subset in list(self.inclusion_json.keys())
        super(OpenImagesO, self).__init__(root=root, transform=transform, target_transform=target_transform,
                                          download=download)

        p = _get_resource_file(self.inclusion_json[subset])
        with open(p, "r") as f:
            included = json.load(f)

        self.files = [join(self.basedir, f) for f in included]


class Places365(iNaturalist):
    """
    Images sourced from the Places365 dataset used as OOD data, usually for CIFAR 10 and 100.
    All labels are -1 by default.

    Dataset set contains 36500 images.

    :see Website: `Places <http://places.csail.mit.edu/browser.html>`__

    .. image:: https://production-media.paperswithcode.com/datasets/Places-0000003475-4b6da14b.jpg
      :target: http://places.csail.mit.edu/browser.html
      :alt: Places 365 examples

    """
    gdrive_id = "1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay"
    filename = "places365.zip"
    target_dir = "places365"
    base_folder = target_dir

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:
        super(Places365, self).__init__(root=root, transform=transform, target_transform=target_transform,
                                          download=download)

        self.files = []

        for d in os.listdir(self.basedir):
            p = join(self.basedir, d)
            if not os.path.isdir(join(p)):
                continue
            self.files += [join(p, f) for f in os.listdir(p)]


if __name__ == "__main__":
    d = Places365(root="/home/ki/datasets/")
    for i in range(len(d)):
        print(d[i])