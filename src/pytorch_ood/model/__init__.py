"""
Models
******************

Publications frequently use the same models, however, hyperparameters, pre-processing differ and
are sometimes cumbersome to set up.

The purpose of this module is to minimize the effort required to reproduce the experiments of others by
providing models, pre-processing and weights, as used in the original publications.


Vision
==================

Wide ResNet
-------------

..  autoclass:: pytorch_ood.model.WideResNet
    :members:


Language
==============================

Models used in pre-LLM papers for OOD detection.

GRU Classifier
---------------------
..  autoclass:: pytorch_ood.model.GRUClassifier
    :members:


Modules
=================
Neural Network modules frequently used in OOD detection.


Class Centers
---------------------
..  autoclass:: pytorch_ood.model.ClassCenters
    :members:


..  autoclass:: pytorch_ood.model.RunningCenters
    :members:


"""

from .centers import ClassCenters, RunningCenters
from .gru import GRUClassifier
from .wrn import WideResNet
