"""
Detectors
******************

This package contains a collection of different Out-of-Distribution Detectors.

API
------
Each detector implements a common API which contains a ``predict`` and a ``fit`` method, where ``fit`` is optional.
The objects ``__call__`` methods become an alias for the ``predict`` function, so you can use

.. code:: python

    detector = Detector(model)
    detector.fit(data_loader)
    scores = detector(x)


Some of the detectors support grid-like input, so that they can be used for anomaly segmentation
without the need for further adjustment.


..  autoclass:: pytorch_ood.api.Detector
    :members:

Maximum Softmax
-------------------------------
.. automodule:: pytorch_ood.detector.softmax

Maximum Logit
-------------------------------
.. automodule:: pytorch_ood.detector.maxlogit

OpenMax
-------------------------------
.. automodule:: pytorch_ood.detector.openmax

ODIN Preprocessing
-------------------------------
.. automodule:: pytorch_ood.detector.odin

Energy Based OOD
-------------------------------
.. automodule:: pytorch_ood.detector.energy

Mahalanobis Method
-------------------------------
.. automodule:: pytorch_ood.detector.mahalanobis

Monte Carlo Dropout
-------------------------------
.. automodule:: pytorch_ood.detector.mcd

Virtual Logit Matching
-------------------------------
.. automodule:: pytorch_ood.detector.vim

KL-Matching
-------------------------------
.. automodule:: pytorch_ood.detector.klmatching

"""
from .energy import EnergyBased
from .klmatching import KLMatching
from .mahalanobis import Mahalanobis
from .maxlogit import MaxLogit
from .mcd import MCD
from .odin import ODIN, odin_preprocessing
from .openmax import OpenMax
from .softmax import MaxSoftmax
from .vim import ViM
