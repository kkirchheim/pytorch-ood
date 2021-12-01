"""

Unsupervised
=====================
Unsupervised losses are only trained on in-distribution data (or similarly, only on
points from known known classes.)

Therefore, all of these loss functions expect that the target labels are strictly > 1.

Confidence Loss
----------------------------------------------
..  autoclass:: oodtk.loss.ConfidenceLoss
    :members:

Class Anchor Clustering Loss
----------------------------------------------
..  autoclass:: oodtk.loss.CACLoss
    :members:

II Loss
----------------------------------------------
..  autoclass:: oodtk.loss.IILoss
    :members:

Center Loss
----------------------------------------------
.. autoclass:: oodtk.loss.CenterLoss
    :members:

Triplet Loss
----------------------------------------------
.. autoclass:: oodtk.loss.TripletLoss
    :members:


Supervised
=====================
Supervised Losses make use from example Out-of-Distribution samples (or samples from known unknown classes).
Thus, these losses can handle samples with target values < 1.

Outlier Exposure
----------------------------------------------
.. autoclass:: oodtk.loss.OutlierExposureLoss
    :members:

Objectosphere
----------------------------------------------
.. autoclass:: oodtk.loss.ObjectosphereLoss
    :members:


Energy Regularized
----------------------------------------------
.. autoclass:: oodtk.loss.EnergyRegularizedLoss
    :members:

"""
from .ii import IILoss
from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .objectosphere import ObjectosphereLoss
from .oe import OutlierExposureLoss
from .triplet import TripletLoss
from .energy import EnergyRegularizedLoss
