"""

Textures
----------------------
..  autoclass:: pytorch_ood.dataset.img.Textures
    :members:

TinyImageNetCrop
----------------------
..  autoclass:: pytorch_ood.dataset.img.TinyImageNetCrop
    :members:

TinyImageNetResize
----------------------
..  autoclass:: pytorch_ood.dataset.img.TinyImageNetResize
    :members:

LSUNCrop
----------------------
..  autoclass:: pytorch_ood.dataset.img.LSUNCrop
    :members:

LSUNResize
----------------------
..  autoclass:: pytorch_ood.dataset.img.LSUNResize
    :members:

80M TinyImages
----------------------
..  autoclass:: pytorch_ood.dataset.img.TinyImages
    :members:

300K Random Images
----------------------
..  autoclass:: pytorch_ood.dataset.img.TinyImages300k
    :members:

Gaussian Noise
----------------------
..  autoclass:: pytorch_ood.dataset.img.GaussianNoise
    :members:

Uniform Noise
----------------------

..  autoclass:: pytorch_ood.dataset.img.UniformNoise
    :members:

MNIST C
----------------------

..  autoclass:: pytorch_ood.dataset.img.MNISTC
    :members:

ImageNet-A
----------------------

..  autoclass:: pytorch_ood.dataset.img.ImageNetA
    :members:

ImageNet-O
----------------------

..  autoclass:: pytorch_ood.dataset.img.ImageNetO
    :members:

ImageNet-C
----------------------

..  autoclass:: pytorch_ood.dataset.img.ImageNetC
    :members:

ImageNet-R
----------------------

..  autoclass:: pytorch_ood.dataset.img.ImageNetR
    :members:

CIFAR10-C
----------------------

..  autoclass:: pytorch_ood.dataset.img.CIFAR10C
    :members:

CIFAR100-C
----------------------

..  autoclass:: pytorch_ood.dataset.img.CIFAR100C
    :members:


StreetHazards
----------------------

..  autoclass:: pytorch_ood.dataset.img.StreetHazards
    :members:

"""
from .cifar import CIFAR10C, CIFAR100C
from .cub200 import Cub2011
from .fooling import FoolingImages
from .imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from .mnistc import MNISTC
from .noise import GaussianNoise, UniformNoise
from .odin import LSUNCrop, LSUNResize, TinyImageNetCrop, TinyImageNetResize
from .stanfordcars import StanfordCars
from .streethazards import StreetHazards
from .textures import Textures
from .tinyimagenet import TinyImagenet
from .tinyimages import TinyImages, TinyImages300k
