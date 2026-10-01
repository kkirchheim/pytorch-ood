"""
Out-of-Distribution detectors and the API they share.

Common Interface
----------------

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
------------------------

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
