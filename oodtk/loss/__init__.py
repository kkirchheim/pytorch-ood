"""

ConfidenceLoss
----------------------------------------------
..  autoclass:: oodtk.loss.ConfidenceLoss
    :members:

IILoss
----------------------------------------------
..  autoclass:: oodtk.loss.IILoss
    :members:

CenterLoss
----------------------------------------------
.. autoclass:: oodtk.loss.CenterLoss
    :members:

TripletLoss
----------------------------------------------
.. autoclass:: oodtk.loss.TripletLoss
    :members:

OutlierExposureLoss
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
