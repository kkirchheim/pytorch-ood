"""
Detectors
******************

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


Feature-based
-------------------------------

ODIN Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.odin


Mahalanobis Distance (MD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_ood.detector.mahalanobis


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

Activation Pruning
---------------------

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
from .energy import EnergyBased
from .entropy import Entropy
from .klmatching import KLMatching
from .mahalanobis import Mahalanobis
from .maxlogit import MaxLogit
from .mcd import MCD
from .odin import ODIN, odin_preprocessing
from .openmax import OpenMax
from .softmax import MaxSoftmax
from .tscaling import TemperatureScaling
from .vim import ViM
from .knn import KNN
from .ash import ASH
from .react import ReAct
from .rmd import RMD
from .dice import DICE
from .she import SHE

