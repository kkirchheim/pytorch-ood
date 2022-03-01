import logging
import typing

import torch
import torch.nn.functional as F
from torch import nn

from oodtk.utils import contains_known, contains_unknown, is_known, is_unknown

log = logging.getLogger(__name__)


class ObjectosphereLoss(nn.Module):
    """
    From *Reducing Network Agnostophobia*.

    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(self, lambda_=1.0, zetta=1.0):
        """"""
        self.lambda_ = lambda_
        self.zetta = zetta

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param logits: class logits
        :param target: target labels
        :return:
        """
        if contains_known(target):
            known = is_known(target)
            loss_known = F.cross_entropy(logits[known], target[known], reduction=None)
            loss_known_sqnorm = None  # TODO
        else:
            loss_known = None

        if contains_unknown(target):
            unknown = is_unknown(target)
            loss_unknown = torch.logsumexp(logits).mean(dim=1)
            loss_unknown_sqnorm = logits.T.dot(logits)
        else:
            loss_unknown = None


class EntropicOpenSetLoss(nn.Module):
    """
    From *Reducing Network Agnostophobia*.

    Uses mean reduction.

    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(self):
        """"""

    def forward(self, logits, target) -> torch.Tensor:
        """

        :param logits: class logits
        :param target: target labels
        :return:
        """
        if contains_known(target):
            known = is_known(target)
            loss_known = F.cross_entropy(logits[known], target[known], reduction=None)
        else:
            loss_known = None

        if contains_unknown(target):
            unknown = is_unknown(target)
            loss_unknown = torch.logsumexp(logits).mean(dim=1)
        else:
            loss_unknown = None

        return torch.stack([loss_known, loss_unknown]).mean()
