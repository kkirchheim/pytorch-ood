"""


Classification
----------------------
Contains datasets often used in anomaly Detection, where the entire input is labels as either ID or OOD.

Textures
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.Textures
    :members:

TinyImageNetCrop
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImageNetCrop
    :members:

TinyImageNetResize
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImageNetResize
    :members:

LSUNCrop
`````````````
..  autoclass:: pytorch_ood.dataset.img.LSUNCrop
    :members:

LSUNResize
`````````````
..  autoclass:: pytorch_ood.dataset.img.LSUNResize
    :members:

TinyImageNet
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImageNet
    :members:

Places365
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.Places365
    :members:

80M TinyImages
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.TinyImages
    :members:

300K Random Images
``````````````````````````

..  autoclass:: pytorch_ood.dataset.img.TinyImages300k
    :members:

ImageNet-A
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetA
    :members:

ImageNet-O
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetO
    :members:

ImageNet-R
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetR
    :members:

ImageNet-V2
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetV2
    :members:

ImageNet-ES
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetES
    :members:

MNIST-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.MNISTC
    :members:

CIFAR10-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.CIFAR10C
    :members:

CIFAR100-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.CIFAR100C
    :members:


CIFAR100-GAN
````````````````````````````
.. autoclass:: pytorch_ood.dataset.img.CIFAR100GAN
    :members:


ImageNet-C
`````````````
..  autoclass:: pytorch_ood.dataset.img.ImageNetC
    :members:

OpenImages-O
`````````````
..  autoclass:: pytorch_ood.dataset.img.OpenImagesO
    :members:

iNaturalist
`````````````
..  autoclass:: pytorch_ood.dataset.img.iNaturalist
    :members:

SSBHard
`````````````
..  autoclass:: pytorch_ood.dataset.img.SSBHard
    :members:


Chars74k
`````````````
..  autoclass:: pytorch_ood.dataset.img.Chars74k
    :members:

Fractals
`````````````
..  autoclass:: pytorch_ood.dataset.img.FractalDataset
    :members:

Fooling Images
````````````````
..  autoclass:: pytorch_ood.dataset.img.FoolingImages
    :members:

NINCO
````````````
..  autoclass:: pytorch_ood.dataset.img.NINCO
    :members:

Feature Visualizations
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.FeatureVisDataset
    :members:

Gaussian Noise
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.GaussianNoise
    :members:

Uniform Noise
`````````````
..  autoclass:: pytorch_ood.dataset.img.UniformNoise
    :members:

Segmentation
----------------------

StreetHazards
`````````````
..  autoclass:: pytorch_ood.dataset.img.StreetHazards
    :members:

FishyScapes
`````````````
..  autoclass:: pytorch_ood.dataset.img.FishyScapes
    :members:

LostAndFound
`````````````
..  autoclass:: pytorch_ood.dataset.img.LostAndFound
    :members:

RoadAnomaly
`````````````
..  autoclass:: pytorch_ood.dataset.img.RoadAnomaly
    :members:

SegmentMeIfYouCan
``````````````````
..  autoclass:: pytorch_ood.dataset.img.SegmentMeIfYouCan
    :members:

MVTech-AD
`````````````
..  autoclass:: pytorch_ood.dataset.img.MVTechAD
    :members:


Object Detection
----------------------

SuMNIST
`````````````
..  autoclass:: pytorch_ood.dataset.img.SuMNIST
    :members:

"""

from .chars74k import Chars74k
from .cifar import CIFAR10C, CIFAR100C
from .fishyscapes import FishyScapes, LostAndFound
from .fooling import FoolingImages
from .goe import CIFAR100GAN
from .imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from .mnistc import MNISTC
from .mvtech import MVTechAD
from .ninco import NINCO
from .noise import GaussianNoise, UniformNoise
from .odin import LSUNCrop, LSUNResize, TinyImageNetCrop, TinyImageNetResize
from .openood import (
    OpenImagesO,
    Places365,
    iNaturalist,
    ImageNetV2,
    ImageNetES,
    SSBHard,
)
from .roadanomaly import RoadAnomaly
from .smiyc import SegmentMeIfYouCan
from .streethazards import StreetHazards
from .sumnist import SuMNIST
from .textures import Textures
from .tinyimagenet import TinyImageNet
from .tinyimages import TinyImages, TinyImages300k
from .pixmix import FeatureVisDataset, FractalDataset
