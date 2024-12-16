"""
Vision
----------------------

Contains augmentations for computer vision tasks.


PixMix
``````````````````````````
..  autoclass:: pytorch_ood.augment.img.PixMixDataset
    :members: __getitem__


InsertCOCO
``````````````````````````
..  autoclass:: pytorch_ood.augment.img.InsertCOCO
    :members: __call__
"""

from .cocopaste import InsertCOCO, COCO
from .pixmix import PixMixDataset
