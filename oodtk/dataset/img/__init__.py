"""

 TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize


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
..  autoclass:: oodtk.dataset.img.RandomImages300K
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
"""
from .cub200 import Cub2011
from .stanfordcars import StanfordCars
from .tinyimagenet import TinyImagenet
from .odin import TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize
from .textures import Textures
from .tinyimages import TinyImages, RandomImages300K
from .noise import GaussianNoise, UniformNoise
from .mnistc import MNISTC
from .imagenet import ImageNetA, ImageNetO

