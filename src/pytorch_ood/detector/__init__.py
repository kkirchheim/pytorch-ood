"""
This module provides a collection of different Out-of-Distribution Detectors.

API
------
Each detector implements a common API which contains a ``predict`` and a ``fit`` method, where ``fit`` is optional.
The objects ``__call__`` methods is delegated to the the ``predict`` function, so you can use

.. code:: python

    detector = Detector(model)
    detector.fit(data_loader)
    scores = detector(x)


Feature-based Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can also use the ``fit_features`` and ``predict_features`` methods.
In that case, inputs will not be passed through the model. This can help to avoid passing
data through the model multiple times when fitting several detectors. Detectors who do not
support this will raise an exception.

.. code:: python

    detector = Detector(model=None)
    detector.fit_features(train_features, train_labels)
    scores = detector.predict_features(test_features)

Some of the detectors support grid-like input, so that they can be used for anomaly segmentation
without further adjustment.


..  autoclass:: pytorch_ood.api.Detector
    :members:


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



Logit-based
-------------------------------



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


Relative Mahalanobis Distance (RMD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.rmd


Virtual Logit Matching (ViM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.vim


Nearest Neighbor (kNN)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.knn


Simplified Hopfield Energy (SHE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.she

Gram Matrices Based (GM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gram


Neural Collapse Inspired (NCI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.nci


Gradient-based
--------------------------

Gradient-based detectors are based on the observation that the gradients (w.r.t. the model parameters or
the inputs) for ID and OOD data behave differently.

GradNorm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.gradnorm


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



"""

from .ash import ASH
from .dice import DICE
from .energy import EnergyBased
from .entropy import Entropy
from .klmatching import KLMatching
from .gram import Gram
from .knn import KNN
from .mahalanobis import Mahalanobis
from .maxlogit import MaxLogit
from .mcd import MCD
from .mmahalanobis import MultiMahalanobis
from .odin import ODIN, odin_preprocessing
from .openmax import OpenMax
from .react import ReAct
from .rmd import RMD
from .she import SHE
from .softmax import MaxSoftmax
from .tscaling import TemperatureScaling
from .vim import ViM
from .webo import WeightedEBO
from .gradnorm import GradNorm
from .nci import NCI
from .nac import NACUE
