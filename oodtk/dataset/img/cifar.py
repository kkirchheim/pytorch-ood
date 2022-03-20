import logging
import os
import sys
from typing import Optional, Callable, Any, Tuple

sys.path.insert(0, os.path.join(os.getcwd(),"oodtk", "dataset", "img"))
from base import ImageDatasetHandler

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class CIFAR10C(ImageDatasetHandler):
    """
    Natural Adversarial Examples

    :see Website: https://zenodo.org/record/2535967

    # https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    """
    base_folder = "CIFAR10C/images/"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"
    tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"


class CIFAR10P(ImageDatasetHandler):
    """
    Natural Adversarial Examples

    :see Website: https://zenodo.org/record/2535967

    # https://zenodo.org/record/2535967/files/CIFAR-10-P.tar
    """
    base_folder = "CIFAR10P/images/"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-P.tar"
    filename = "CIFAR-10-P.tar"
    tgz_md5 = "125d6775afc5846ea84584d7524dedff"


class CIFAR100C(ImageDatasetHandler):
    """
    Natural Adversarial Examples

    :see Website: https://zenodo.org/record/3555552

    # https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    """
    base_folder = "CIFAR100C/images/"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
    tgz_md5 = "11f0ed0f1191edbf9fa23466ae6021d3 "


