import logging
import typing

import torch
from torch import nn
import torch.nn.functional as F
from oodtk import utils


log = logging.getLogger(__name__)


class ObjectosphereLoss(nn.Module):
    """
    Objectosphere Loss
    """
