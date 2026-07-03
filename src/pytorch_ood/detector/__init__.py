"""
This module provides a collection of different Out-of-Distribution Detectors.

API
------
Each detector implements a common API which contains a ``predict`` and a ``fit`` method, where ``fit`` is optional.
The objects ``__call__`` methods is delegated to the ``predict`` function, so you can use

.. code:: python

    detector = Detector(model)
    detector.fit(data_loader)
    scores = detector(x)


..  autoclass:: pytorch_ood.api.Detector
    :members:



Some of the detectors support grid-like input, so that they can be used for anomaly segmentation
without further adjustment.


Representation Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, detectors can be used on intermediate representations without passing inputs
through the full model again. The available methods will depend on the base class of the detector:

- logits detectors: ``predict_logits(...)`` and optionally ``fit_logits(...)``
- feature detectors: ``predict_features(...)`` and optionally ``fit_features(...)``
- feature-map detectors: ``predict_feature_maps(...)`` and optionally
  ``fit_feature_maps(...)``
- structured detectors: ``predict_structured(...)`` and optionally
  ``fit_structured(...)``

.. code:: python

    detector = LogitsDetector(model=None)
    detector.fit_logits(train_logits, train_labels)
    scores = detector.predict_logits(test_logits)


..  autoclass:: pytorch_ood.api.LogitsDetector
    :members:
    :show-inheritance:

..  autoclass:: pytorch_ood.api.FeaturesDetector
    :members:
    :show-inheritance:

..  autoclass:: pytorch_ood.api.FeatureMapsDetector
    :members:
    :show-inheritance:

..  autoclass:: pytorch_ood.api.StructuredDetector
    :members:
    :show-inheritance:

..  autoclass:: pytorch_ood.api.GradientDetector
    :members:
    :show-inheritance:


Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


..  inheritance-diagram:: pytorch_ood.api.Detector pytorch_ood.api.LogitsDetector pytorch_ood.api.FeaturesDetector pytorch_ood.api.FeatureMapsDetector pytorch_ood.api.StructuredDetector pytorch_ood.api.GradientDetector pytorch_ood.detector.MaxSoftmax pytorch_ood.detector.TemperatureScaling pytorch_ood.detector.Entropy pytorch_ood.detector.KLMatching pytorch_ood.detector.GEN pytorch_ood.detector.MCD pytorch_ood.detector.MaxLogit pytorch_ood.detector.OpenMax pytorch_ood.detector.EnergyBased pytorch_ood.detector.WeightedEBO pytorch_ood.detector.Mahalanobis pytorch_ood.detector.MahalanobisODIN pytorch_ood.detector.RMD pytorch_ood.detector.ViM pytorch_ood.detector.KNN pytorch_ood.detector.NNGuide pytorch_ood.detector.SHE pytorch_ood.detector.Gram pytorch_ood.detector.NCI pytorch_ood.detector.fDBD pytorch_ood.detector.GMM pytorch_ood.detector.MCM pytorch_ood.detector.PNML pytorch_ood.detector.GradNorm pytorch_ood.detector.GradNormKL pytorch_ood.detector.ODIN pytorch_ood.detector.MCD pytorch_ood.detector.ASH pytorch_ood.detector.ReAct pytorch_ood.detector.DICE pytorch_ood.detector.RankFeat pytorch_ood.detector.VRA pytorch_ood.detector.SCALE pytorch_ood.detector.MultiMahalanobis pytorch_ood.detector.NACUE
    :parts: 1
    :top-classes: pytorch_ood.api.Detector


Probability-based
-------------------------------

Probability-based methods are based on the observation that OOD inputs tend to be assigned lower posteriors with higher
entropy, i.e., the predicted distribution is often less concentrated on a single class.


Maximum Softmax (MSP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.softmax

Monte Carlo Dropout (MCD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.mcd

Temperature Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.tscaling

KL-Matching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.klmatching

Entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.entropy

Generalized Entropy (GEN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gen



Logit-based
-------------------------------

Logit-based methods are based on the observation that OOD inputs tend to yield different logits compared to ID data.

Maximum Logit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.maxlogit

OpenMax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.openmax

Energy Based (EBO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.energy

Weighted Energy Based (WEBO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.webo



Feature-based
-------------------------------


Mahalanobis Distance (MD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.mahalanobis

Multi-Layer Mahalanobis Distance (MD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.mmahalanobis

Mahalanobis Distance with ODIN (MahalanobisODIN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pytorch_ood.detector.MahalanobisODIN
    :members:
    :inherited-members:
    :show-inheritance:


Relative Mahalanobis Distance (RMD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.rmd


Virtual Logit Matching (ViM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.vim


Nearest Neighbor (kNN)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
   ``pytorch_ood.detector.KNN`` requires ``scikit-learn`` to be installed.

.. automodule:: pytorch_ood.detector.knn

Nearest Neighbor Guidance (NNGuide)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.nnguide


Simplified Hopfield Energy (SHE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.she

Gram Matrices Based (GM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gram


Neural Collapse Inspired (NCI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.nci

Fast Decision Boundary Distance (fDBD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.fdbd

Gaussian Mixture Model (GMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gmm

Predictive Normalized Maximum Likelihood (pNML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.pnml

Maximum Concept Matching (MCM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.mcm

Logit Scaling (LTS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.lts


Gradient-based
--------------------------

Gradient-based detectors are based on the observation that the gradients (w.r.t. the model parameters or
the inputs) for ID and OOD data behave differently. All gradient-based detectors inherit from
:class:`pytorch_ood.api.GradientDetector`.

GradNorm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gradnorm

GradNormKL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gradnormkl


ODIN Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.odin


NAC-UE
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.nac



Activation Pruning
---------------------

Activation pruning methods are based on the observation that OOD inputs cause unusual activations in the model,
and that, by rectifying these unusual activations, we can often improve discriminability of ID and OOD samples.


Activation Shaping (ASH)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.ash

ReAct
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.react

DICE
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.dice

RankFeat
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.rankfeat

VRA
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.vra

SCALE
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.scale


"""

from .ash import ASH
from .dice import DICE
from .energy import EnergyBased
from .entropy import Entropy
from .fdbd import fDBD
from .gen import GEN
from .gmm import GMM
from .gradnorm import GradNorm
from .gradnormkl import GradNormKL
from .gram import Gram
from .klmatching import KLMatching
from .knn import KNN
from .lts import LTS
from .mahalanobis import Mahalanobis, MahalanobisODIN
from .maxlogit import MaxLogit
from .mcd import MCD
from .mcm import MCM
from .mmahalanobis import MultiMahalanobis
from .nac import NACUE
from .nci import NCI
from .nnguide import NNGuide
from .odin import ODIN, odin_preprocessing
from .openmax import OpenMax
from .pnml import PNML
from .rankfeat import RankFeat
from .react import ReAct
from .rmd import RMD
from .scale import SCALE
from .she import SHE
from .softmax import MaxSoftmax
from .tscaling import TemperatureScaling
from .vim import ViM
from .vra import VRA
from .webo import WeightedEBO

__all__ = [
    "ASH",
    "DICE",
    "EnergyBased",
    "Entropy",
    "fDBD",
    "GEN",
    "GMM",
    "GradNorm",
    "GradNormKL",
    "Gram",
    "KLMatching",
    "KNN",
    "LTS",
    "Mahalanobis",
    "MahalanobisODIN",
    "MaxLogit",
    "MaxSoftmax",
    "MCD",
    "MCM",
    "MultiMahalanobis",
    "NACUE",
    "NCI",
    "NNGuide",
    "ODIN",
    "odin_preprocessing",
    "OpenMax",
    "PNML",
    "RMD",
    "RankFeat",
    "ReAct",
    "SCALE",
    "SHE",
    "TemperatureScaling",
    "ViM",
    "VRA",
    "WeightedEBO",
]
