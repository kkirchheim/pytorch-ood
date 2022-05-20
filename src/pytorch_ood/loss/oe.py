import logging

import torch
from torch import nn

from ..utils import apply_reduction, contains_unknown, is_unknown
from .crossentropy import cross_entropy

log = logging.getLogger(__name__)


class OutlierExposureLoss(nn.Module):
    """
    From the paper *Deep Anomaly Detection With Outlier Exposure*.

    In addition to the cross-entropy for known samples, includes an :math:`\\mathcal{L}_{OE}` term
    for OOD samples that is defined as:

    .. math::  \\mathcal{L}_{OE}(x_{out}) =  - \\alpha (\\sum_y f(x_{out})_y - \\log(\\sum_y e^{f(x_{out})_y}))

    which is the cross-entropy between the predicted distribution and the uniform distribution.

    :see Paper: https://arxiv.org/pdf/1812.04606v1.pdf
    """

    def __init__(self, alpha=0.5, reduction="mean"):
        """

        :param alpha: weighting coefficient
        :param reduction: reduction to apply
        """
        super(OutlierExposureLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target) -> torch.Tensor:
        """

        :param logits: class logits for predictions
        :param target: labels for predictions
        :return: loss
        """
        loss_oe = torch.zeros(logits.shape[0], device=logits.device)
        loss_ce = cross_entropy(logits, target, reduction="none")

        if contains_unknown(target):
            unknown = is_unknown(target)
            loss_oe[unknown] = -(
                logits[unknown].mean(dim=1) - torch.logsumexp(logits[unknown], dim=1)
            )
        else:
            loss_oe = 0

        return apply_reduction(loss_ce + self.alpha * loss_oe, reduction=self.reduction)
