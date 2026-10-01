"""
Publications frequently use the same models, however, hyperparameters, pre-processing differ and
are sometimes cumbersome to set up.

The purpose of this module is to minimize the effort required to reproduce the experiments of others by
providing models, pre-processing and weights, as used in the original publications.
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
from .resnet import ResNet18, ResNet50
from .wrn import WideResNet
