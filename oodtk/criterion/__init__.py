"""
Loss functions used in OOD/OSR, implemented as ``torch.nn.Module``.

ConfidenceLoss
----------------------------------------------
..  autoclass:: oodtk.criterion.ConfidenceLoss
    :members:


CACLoss
----------------------------------------------

..  autoclass:: oodtk.criterion.CACLoss
    :members:

..  automodule:: oodtk.criterion.cac
    :members:
        rejection_score

IILoss
----------------------------------------------
..  autoclass:: oodtk.criterion.IILoss
    :members:

CenterLoss
----------------------------------------------
.. autoclass:: oodtk.criterion.CenterLoss
    :members:

TripletLoss
----------------------------------------------
.. autoclass:: oodtk.criterion.TripletLoss
    :members:

OutlierExposureLoss
----------------------------------------------
.. autoclass:: oodtk.criterion.OutlierExposureLoss
    :members:

"""

from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .ii import IILoss
from .triplet import TripletLoss
from .oe import OutlierExposureLoss
