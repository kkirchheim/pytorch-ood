import logging
from typing import Optional

import torch
from torch import nn

from ..utils import apply_reduction, contains_unknown, is_unknown
from .crossentropy import cross_entropy

log = logging.getLogger(__name__)


class OutlierExposureLoss(nn.Module):
    """
    Loss from the paper *Deep Anomaly Detection With Outlier Exposure*.
    While the formulation in the original paper is very general, this module implements the exact loss that
    was used in the corresponding experiments.

    The loss is defined as

    .. math::
        \\mathcal{L}(x, y)
       =
       \\Biggl \\lbrace
       {
       -\\log \\sigma_y(f(x)) \\quad \\quad \\quad  \\quad   \\quad \\quad \\quad  \\quad  \\quad \\quad  \\text{if } y \\geq 0
        \\atop
       \\alpha (\\sum_{c=1}^C f(x)_c - \\log(\\sum_{c=1}^C  e^{f(x)_c})) \\quad \\text{ otherwise }
       }


    where :math:`C` is the number of classes, :math:`\\alpha` is a hyper parameter, and :math:`\\sigma_y`
    denotes the :math:`y^{th}` softmax output.

    :see Paper: `ArXiv <https://arxiv.org/pdf/1812.04606v1.pdf>`__
    :see Implementation: `GitHub <https://github.com/hendrycks/outlier-exposure>`__
    """

    def __init__(self, alpha: float = 0.5, reduction: Optional[str] = "mean"):
        """

        :param alpha: weighting coefficient :math:`\\alpha`
        :param reduction: reduction method, one of ``mean``, ``sum`` or ``none``
        """
        super(OutlierExposureLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param logits: class logits for predictions
        :param target: labels for predictions
        :return: loss
        """
        loss_oe = torch.zeros(logits.shape[0], device=logits.device)
        loss_ce = cross_entropy(logits, target, reduction=None)

        if contains_unknown(target):
            unknown = is_unknown(target)
            loss_oe[unknown] = -(
                logits[unknown].mean(dim=1) - torch.logsumexp(logits[unknown], dim=1)
            )

        return apply_reduction(loss_ce + self.alpha * loss_oe, reduction=self.reduction)
