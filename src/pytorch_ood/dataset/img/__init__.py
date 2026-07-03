"""


Classification
----------------------
Contains datasets often used in anomaly Detection, where the entire input is labels as either ID or OOD.

Textures
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.Textures
    :members:
    :no-index:

TinyImageNetCrop
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImageNetCrop
    :members:
    :no-index:

TinyImageNetResize
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImageNetResize
    :members:
    :no-index:

LSUNCrop
`````````````
..  autoclass:: pytorch_ood.dataset.img.LSUNCrop
    :members:
    :no-index:

LSUNResize
`````````````
..  autoclass:: pytorch_ood.dataset.img.LSUNResize
    :members:
    :no-index:

TinyImageNet
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImageNet
    :members:
    :no-index:

Places365
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.Places365
    :members:
    :no-index:

80M TinyImages
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImages
    :members:
    :no-index:

300K Random Images
``````````````````````````

..  autoclass:: pytorch_ood.dataset.img.TinyImages300k
    :members:
    :no-index:

ImageNet-A
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetA
    :members:
    :no-index:

ImageNet-O
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetO
    :members:
    :no-index:

ImageNet-R
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetR
    :members:
    :no-index:

ImageNet-200
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNet200
    :members:
    :no-index:

ImageNet-V2
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetV2
    :members:
    :no-index:

ImageNet-ES
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetES
    :members:
    :no-index:

MNIST-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.MNISTC
    :members:
    :no-index:

CIFAR10-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.CIFAR10C
    :members:
    :no-index:

CIFAR100-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.CIFAR100C
    :members:
    :no-index:


CIFAR100-GAN
````````````````````````````
.. autoclass:: pytorch_ood.dataset.img.CIFAR100GAN
    :members:
    :no-index:


ImageNet-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetC
    :members:
    :no-index:

OpenImages-O
`````````````
..  autoclass:: pytorch_ood.dataset.img.OpenImagesO
    :members:
    :no-index:

iNaturalist
`````````````
..  autoclass:: pytorch_ood.dataset.img.iNaturalist
    :members:
    :no-index:

SSBHard
`````````````
..  autoclass:: pytorch_ood.dataset.img.SSBHard
    :members:
    :no-index:


Chars74k
`````````````
..  autoclass:: pytorch_ood.dataset.img.Chars74k
    :members:
    :no-index:

Fractals
`````````````
..  autoclass:: pytorch_ood.dataset.img.FractalDataset
    :members:
    :no-index:

Fooling Images
````````````````
..  autoclass:: pytorch_ood.dataset.img.FoolingImages
    :members:
    :no-index:

NINCO
````````````
..  autoclass:: pytorch_ood.dataset.img.NINCO
    :members:
    :no-index:

Feature Visualizations
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.FeatureVisDataset
    :members:
    :no-index:

Gaussian Noise
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.GaussianNoise
    :members:
    :no-index:

Uniform Noise
`````````````
..  autoclass:: pytorch_ood.dataset.img.UniformNoise
    :members:
    :no-index:

Segmentation
----------------------

StreetHazards
`````````````
..  autoclass:: pytorch_ood.dataset.img.StreetHazards
    :members:
    :no-index:

FishyScapes
`````````````
..  autoclass:: pytorch_ood.dataset.img.FishyScapes
    :members:
    :no-index:

LostAndFound
`````````````
..  autoclass:: pytorch_ood.dataset.img.LostAndFound
    :members:
    :no-index:

RoadAnomaly
`````````````
..  autoclass:: pytorch_ood.dataset.img.RoadAnomaly
    :members:
    :no-index:

SegmentMeIfYouCan
``````````````````
..  autoclass:: pytorch_ood.dataset.img.SegmentMeIfYouCan
    :members:
    :no-index:

MVTech-AD
`````````````
..  autoclass:: pytorch_ood.dataset.img.MVTechAD
    :members:
    :no-index:


Object Detection
----------------------

SuMNIST
`````````````
..  autoclass:: pytorch_ood.dataset.img.SuMNIST
    :members:
    :no-index:


Generic
----------------------

ImageListDataset
`````````````````````
..  autoclass:: pytorch_ood.dataset.img.ImageListDataset
    :members:
    :no-index:

"""

from .chars74k import Chars74k
from .cifar import CIFAR10C, CIFAR100C
from .fishyscapes import FishyScapes, LostAndFound
from .fooling import FoolingImages
from .goe import CIFAR100GAN
from .imagelist import ImageListDataset
from .imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from .imagenet200 import ImageNet200
from .mnistc import MNISTC
from .mvtech import MVTechAD
from .ninco import NINCO
from .noise import GaussianNoise, UniformNoise
from .odin import LSUNCrop, LSUNResize, TinyImageNetCrop, TinyImageNetResize
from .openood import (
    ImageNetES,
    ImageNetV2,
    OpenImagesO,
    Places365,
    SSBHard,
    iNaturalist,
)
from .pixmix import FeatureVisDataset, FractalDataset
from .roadanomaly import RoadAnomaly
from .smiyc import SegmentMeIfYouCan
from .streethazards import StreetHazards
from .sumnist import SuMNIST
from .textures import Textures
from .tinyimagenet import TinyImageNet
from .tinyimages import TinyImages, TinyImages300k

__all__ = [
    "Chars74k",
    "ImageListDataset",
    "CIFAR10C",
    "CIFAR100C",
    "FishyScapes",
    "LostAndFound",
    "FoolingImages",
    "CIFAR100GAN",
    "ImageNetA",
    "ImageNetC",
    "ImageNetO",
    "ImageNetR",
    "ImageNet200",
    "MNISTC",
    "MVTechAD",
    "NINCO",
    "GaussianNoise",
    "UniformNoise",
    "LSUNCrop",
    "LSUNResize",
    "TinyImageNetCrop",
    "TinyImageNetResize",
    "OpenImagesO",
    "Places365",
    "iNaturalist",
    "ImageNetV2",
    "ImageNetES",
    "SSBHard",
    "RoadAnomaly",
    "SegmentMeIfYouCan",
    "StreetHazards",
    "SuMNIST",
    "Textures",
    "TinyImageNet",
    "TinyImages",
    "TinyImages300k",
    "FeatureVisDataset",
    "FractalDataset",
]
