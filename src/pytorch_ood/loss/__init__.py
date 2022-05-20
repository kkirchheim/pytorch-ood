"""
Unsupervised
=====================
Unsupervised losses are only trained on in-distribution data (or similarly, only on
points from known known classes.)

Therefore, all of these loss functions expect that the target labels are strictly :math:`\\geq 0`.

Confidence Loss
----------------------------------------------
..  autoclass:: pytorch_ood.loss.ConfidenceLoss
    :members:

Class Anchor Clustering Loss
----------------------------------------------
..  autoclass:: pytorch_ood.loss.CACLoss
    :members:

II Loss
----------------------------------------------
..  autoclass:: pytorch_ood.loss.IILoss
    :members:

Center Loss
----------------------------------------------
.. autoclass:: pytorch_ood.loss.CenterLoss
    :members:


Deep SVDD
----------------------------------------------
.. autoclass:: pytorch_ood.loss.DeepSVDD
    :members:


Entropic Open-Set Loss
----------------------------------------------
.. autoclass:: pytorch_ood.loss.EntropicOpenSetLoss
    :members:


Supervised
=====================
Supervised Losses make use from example Out-of-Distribution samples (or samples from known unknown classes).
Thus, these losses can handle samples with target values :math:`< 0`.

Outlier Exposure
----------------------------------------------
.. autoclass:: pytorch_ood.loss.OutlierExposureLoss
    :members:

Objectosphere
----------------------------------------------
.. autoclass:: pytorch_ood.loss.ObjectosphereLoss
    :members:


Energy Regularized
----------------------------------------------
.. autoclass:: pytorch_ood.loss.EnergyRegularizedLoss
    :members:


Background Class
----------------------------------------------
.. autoclass:: pytorch_ood.loss.BackgroundClassLoss
    :members:

"""
from .background import BackgroundClassLoss
from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .crossentropy import CrossEntropy
from .energy import EnergyRegularizedLoss
from .ii import IILoss
from .objectosphere import EntropicOpenSetLoss, ObjectosphereLoss
from .oe import OutlierExposureLoss

# from .triplet import TripletLoss
from .svdd import DeepSVDD
