"""
Models
******************

Publications frequently use the same models, however, hyperparameters, pre-processing differ and
are sometimes cumbersome to set up.

The purpose of this module is to minimize the effort required to reproduce the experiments of others by
providing models, pre-processing and weights, as used in the original publications.


Pre-Trained Models
==================

Pre-trained models are identified by a string of the form ``arch/dataset/loss/seed``
(the seed is omitted when unknown) and can be loaded with a single call:

..  code-block:: python

    from pytorch_ood.model import load_model, load_transform

    model = load_model("wrn-40-2/cifar10/logitnorm/s0")
    transform = load_transform("wrn-40-2/cifar10/logitnorm/s0")


.. include:: generated/pretrained_models.rst


..  autofunction:: pytorch_ood.model.load_model

..  autofunction:: pytorch_ood.model.load_transform

..  autofunction:: pytorch_ood.model.get_model_info

..  autofunction:: pytorch_ood.model.list_models

..  autoclass:: pytorch_ood.model.ModelEntry
    :members:

..  autoclass:: pytorch_ood.model.ImagePreprocessing
    :members:




Vision
==================

Wide ResNet
-------------

..  autoclass:: pytorch_ood.model.WideResNet
    :members:


ResNet-18
-------------

..  autoclass:: pytorch_ood.model.ResNet18
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
from .registry import (
    ImagePreprocessing,
    ModelEntry,
    Preprocessing,
    get_model_info,
    list_models,
    load_model,
    load_transform,
)
from .resnet import ResNet18
from .wrn import WideResNet
