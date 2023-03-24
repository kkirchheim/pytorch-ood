"""

All objective functions are implemented as ``torch.nn.Modules``. Integrating custom reduction for all of them is
work in progress.


Unsupervised
=====================
Unsupervised losses are only trained on in-distribution data (or similarly, only on
points from known known classes.)

Therefore, all of these loss functions expect that the target labels are strictly :math:`\\geq 0`.


Deep SVDD Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge


.. autoclass:: pytorch_ood.loss.DeepSVDDLoss
    :members:


Class Anchor Clustering Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge


..  autoclass:: pytorch_ood.loss.CACLoss
    :members:


II Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge


..  autoclass:: pytorch_ood.loss.IILoss
    :members:


Center Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge


.. autoclass:: pytorch_ood.loss.CenterLoss
    :members:


Cross-Entropy Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.loss.CrossEntropyLoss
    :members:


Confidence Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.loss.ConfidenceLoss
    :members:



Supervised
=====================
Supervised Losses make use from example Out-of-Distribution samples (or samples from known unknown classes).
Thus, these losses can handle samples with target values :math:`< 0`.


Outlier Exposure Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

.. autoclass:: pytorch_ood.loss.OutlierExposureLoss
    :members:


Entropic Open-Set Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: classification badge

.. autoclass:: pytorch_ood.loss.EntropicOpenSetLoss
    :members:


Objectosphere Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

.. autoclass:: pytorch_ood.loss.ObjectosphereLoss
    :members:


Energy-Bounded Learning Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

.. autoclass:: pytorch_ood.loss.EnergyRegularizedLoss
    :members:


MCHAD Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.loss.MCHADLoss
    :members:



Background Class Loss
----------------------------------------------

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

.. autoclass:: pytorch_ood.loss.BackgroundClassLoss
    :members:

"""
from .background import BackgroundClassLoss
from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .crossentropy import CrossEntropyLoss
from .energy import EnergyRegularizedLoss
from .entropy import EntropicOpenSetLoss
from .ii import IILoss
from .mchad import MCHADLoss
from .objectosphere import ObjectosphereLoss
from .oe import OutlierExposureLoss

# from .triplet import TripletLoss
from .svdd import DeepSVDDLoss, SSDeepSVDDLoss
