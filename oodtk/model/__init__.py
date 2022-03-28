"""
Models
==================
Frequently used Neural Network based Models

Wide ResNet
-------------

..  autoclass:: oodtk.model.WideResNet
    :members:

Vision Transformer
-------------
..  autoclass:: oodtk.model.VisionTransformer
    :members:

Pre-Trained
-------------

Wide ResNet
+++++++++++++++++++++++

..  autoclass:: oodtk.model.WideResNetPretrained
    :members:

"""
from .vit import VisionTransformer
from .wrn import WideResNet, WideResNetPretrained
