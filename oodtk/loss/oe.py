import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from oodtk.utils import contains_known, contains_unknown, is_known, is_unknown

from .crossentropy import cross_entropy

log = logging.getLogger(__name__)


class OutlierExposureLoss(nn.Module):
    """
    From the paper *Deep Anomaly Detection With Outlier Exposure*.

    The loss for OOD samples is the cross-entropy between the predicted distribution and the uniform distribution.

    .. math:: \\sum_{x,y \\in \\mathcal{D}^{in}} \\mathcal{L}_{NLL}(f(x),y) + \\lambda
        \\sum_{x \\in \\mathcal{D}^{out}} D_{KL}(f(x) \\Vert \\mathcal{U})

    :see Paper: https://arxiv.org/pdf/1812.04606v1.pdf
    """

    def __init__(self, num_classes, lmbda=0.5):
        """

        :param lmbda: weighting coefficient
        """
        super(OutlierExposureLoss, self).__init__()
        self.lambda_ = lmbda

    def forward(self, logits, target) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param logits: class logits for predictions
        :param target: labels for predictions
        :return: tuple with cross-entropy for known samples and weighted outlier exposure loss for unknown samples.
        """
        loss_ce = cross_entropy(logits, target)
        if contains_unknown(target):
            unknown = is_unknown(target)
            loss_oe = -(logits[unknown].mean(1) - torch.logsumexp(logits[unknown], dim=1)).mean()
        else:
            loss_oe = 0
        return loss_ce, self.lambda_ * loss_oe
