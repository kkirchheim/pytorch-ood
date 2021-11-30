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

TinyImages
----------------------
..  autoclass:: oodtk.dataset.img.TinyImages
    :members:
"""
from .cub200 import Cub2011
from .stanfordcars import StanfordCars
from .tinyimagenet import TinyImagenet
from .odin import TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize
from .textures import Textures
from .tinyimages import TinyImages
