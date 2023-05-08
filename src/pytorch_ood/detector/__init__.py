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
^^^^^^^^^^^^^^^^^^^^^^^^

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

Maximum Softmax (MSP)
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

Energy Based
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

Entropy
-------------------------------
.. automodule:: pytorch_ood.detector.entropy


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
from .vim import ViM
