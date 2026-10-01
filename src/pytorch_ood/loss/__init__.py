"""
All objective functions are implemented as ``torch.nn.Modules``.
Some of them have a set of trainable parameters and must be moved to the appropriate device.
"""

import warnings

from .background import BackgroundClassLoss
from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .crossentropy import CrossEntropyLoss
from .energy import EnergyRegularizedLoss
from .entropy import EntropicOpenSetLoss
from .ii import IILoss
from .logitnorm import LogitNorm, logit_norm_loss
from .mchad import MCHADLoss
from .objectosphere import ObjectosphereLoss
from .oe import OutlierExposureLoss
from .scone import EnergyMarginLoss

# from .triplet import TripletLoss
from .svdd import DeepSADLoss, DeepSVDDLoss
from .vos import VirtualOutlierSynthesizingRegLoss, VOSRegLoss


def __getattr__(name):
    # renamed; the old name keeps working for now
    if name == "SSDeepSVDDLoss":
        warnings.warn(
            "SSDeepSVDDLoss was renamed to DeepSADLoss and will be removed in a future release",
            DeprecationWarning,
            stacklevel=2,
        )
        return DeepSADLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
