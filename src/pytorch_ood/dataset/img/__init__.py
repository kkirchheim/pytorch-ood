"""


Classification
----------------------
Contains datasets often used in anomaly Detection, where the entire input is labels as either IN or OOD.

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

Chars74k
`````````````
..  autoclass:: pytorch_ood.dataset.img.Chars74k
    :members:


PixMix
``````````````````````````
..  autoclass:: pytorch_ood.dataset.img.PixMixDataset
    :members:


Fractals
`````````````
..  autoclass:: pytorch_ood.dataset.img.FractalDataset
    :members:


Fooling Images
````````````````
..  autoclass:: pytorch_ood.dataset.img.FoolingImages
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

MVTech-AD
`````````````
..  autoclass:: pytorch_ood.dataset.img.MVTechAD
    :members:

"""
from .chars74k import Chars74k
from .cifar import CIFAR10C, CIFAR100C
from .fooling import FoolingImages
from .imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetR
from .mnistc import MNISTC
from .mvtech import MVTechAD
from .noise import GaussianNoise, UniformNoise
from .odin import LSUNCrop, LSUNResize, TinyImageNetCrop, TinyImageNetResize
from .pixmix import FeatureVisDataset, FractalDataset, PixMixDataset
from .streethazards import StreetHazards
from .textures import Textures
from .tinyimagenet import TinyImageNet
from .tinyimages import TinyImages, TinyImages300k
from .openood import OpenImagesO, iNaturalist, Places365
