"""

Unsupervised
=====================

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

Outlier Exposure
----------------------------------------------
.. autoclass:: oodtk.loss.OutlierExposureLoss
    :members:

Objectosphere
----------------------------------------------
.. autoclass:: oodtk.loss.ObjectosphereLoss
    :members:

"""
from .ii import IILoss
from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .objectosphere import ObjectosphereLoss
from .oe import OutlierExposureLoss
from .triplet import TripletLoss
