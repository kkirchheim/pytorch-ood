"""

Textures
----------------------
..  autoclass:: oodtk.dataset.img.Textures
    :members:

TinyImageNetCrop
----------------------
..  autoclass:: oodtk.dataset.img.TinyImageNetCrop
    :members:

TinyImageNetResize
----------------------
..  autoclass:: oodtk.dataset.img.TinyImageNetResize
    :members:

LSUNCrop
----------------------
..  autoclass:: oodtk.dataset.img.LSUNCrop
    :members:

LSUNResize
----------------------
..  autoclass:: oodtk.dataset.img.LSUNResize
    :members:

80M TinyImages
----------------------
..  autoclass:: oodtk.dataset.img.TinyImages
    :members:

300K Random Images
----------------------
..  autoclass:: oodtk.dataset.img.TinyImages300k
    :members:

Gaussian Noise
----------------------
..  autoclass:: oodtk.dataset.img.GaussianNoise
    :members:

Uniform Noise
----------------------

..  autoclass:: oodtk.dataset.img.UniformNoise
    :members:

MNIST C
----------------------

..  autoclass:: oodtk.dataset.img.MNISTC
    :members:

ImageNet-A
----------------------

..  autoclass:: oodtk.dataset.img.ImageNetA
    :members:

ImageNet-O
----------------------

..  autoclass:: oodtk.dataset.img.ImageNetO
    :members:

ImageNet-C
----------------------

..  autoclass:: oodtk.dataset.img.ImageNetC
    :members:

ImageNet-R
----------------------

..  autoclass:: oodtk.dataset.img.ImageNetR
    :members:

StreetHazards
----------------------

..  autoclass:: oodtk.dataset.img.StreetHazards
    :members:

"""
from .cub200 import Cub2011
from .imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetP, ImageNetR
from .mnistc import MNISTC
from .noise import GaussianNoise, UniformNoise
from .odin import LSUNCrop, LSUNResize, TinyImageNetCrop, TinyImageNetResize
from .stanfordcars import StanfordCars
from .streethazards import StreetHazards
from .textures import Textures
from .tinyimagenet import TinyImagenet
from .tinyimages import TinyImages, TinyImages300k
