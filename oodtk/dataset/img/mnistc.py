import logging
import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

log = logging.getLogger(__name__)


class MNISTC(VisionDataset):
    """
    MNIST-C is MNIST with corruptions for benchmarking OOD methods.

    :see Paper: https://arxiv.org/pdf/1906.02337.pdf
    :see Download: https://zenodo.org/record/3239543
    """

    url = "https://zenodo.org/record/3239543/files/mnist_c.zip?download=1"
