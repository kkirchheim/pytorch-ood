"""
Detectors
******************
Out-of-Distribution Detectors


Softmax
-------------------------------

.. automodule:: pytorch_ood.detector.softmax

MaxLogit
-------------------------------

.. automodule:: pytorch_ood.detector.maxlogit

OpenMax
-------------------------------

.. automodule:: pytorch_ood.detector.openmax


ODIN
-------------------------------

.. automodule:: pytorch_ood.detector.odin

Negative Energy
-------------------------------

.. automodule:: pytorch_ood.detector.energy

Mahalanobis Distance
-------------------------------

.. automodule:: pytorch_ood.detector.mahalanobis

Monte Carlo Dropout
-------------------------------

.. automodule:: pytorch_ood.detector.mcd
"""
from .energy import NegativeEnergy
from .mahalanobis import Mahalanobis
from .maxlogit import MaxLogit
from .mcd import MCD
from .odin import ODIN
from .openmax import OpenMax
from .softmax import Softmax
