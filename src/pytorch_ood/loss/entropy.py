import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import apply_reduction, contains_known, contains_unknown, is_known, is_unknown
from .crossentropy import cross_entropy


class EntropicOpenSetLoss(nn.Module):
    """
    From the paper *Reducing Network Agnostophobia*.
    The loss aims to maximizes the entropy for OOD inputs.

    A variant for segmentation was proposed in
    *Entropy Maximization and Meta Classification for Out-Of-Distribution Detection in Semantic Segmentation*.

    The loss is calculated as

    .. math::
       \\mathcal{L}(x, y)
       =
       \\Biggl \\lbrace
       {
       -\\log \\sigma_y(f(x)) \\quad \\text{if } y \\geq 0
        \\atop
       \\frac{1}{C} \\sum_{c=1}^C \\log \\sigma_c(f(x)) \\quad \\text{ otherwise }
       }

    where :math:`\\sigma` is the softmax function and :math:`C` is the number of classes.


    :see Paper:
        `NeurIPS <https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf>`__
    :see Paper:
        `ArXiv <https://arxiv.org/pdf/2012.06575.pdf>`__
    """

    def __init__(self, reduction: Optional[str] = "mean"):
        """
        :param reduction: reduction method, one of ``mean``, ``sum`` or ``none``
        """
        super(EntropicOpenSetLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """

        :param logits: class logits
        :param target: target labels
        :return: the loss
        """
        if len(logits.shape) == 2:
            losses = torch.zeros(size=(logits.shape[0],)).to(logits.device)

            # default
            if contains_known(target):
                known = is_known(target)
                losses[known] = F.cross_entropy(logits[known], target[known], reduction="none")

            if contains_unknown(target):
                unknown = is_unknown(target)
                losses[unknown] = -F.log_softmax(logits[unknown], dim=1).mean(dim=1)

            return apply_reduction(losses, self.reduction)
        elif len(logits.shape) == 4:
            losses_in = cross_entropy(logits, target, reduction="none")
            losses_out = -F.log_softmax(logits, dim=1).mean(dim=1)
            # older torch versions need explicit single precision float here
            fp32zero = torch.zeros((1,), dtype=torch.float, device=logits.device)
            losses_out = torch.where(target.float() < 0, losses_out, fp32zero)
            losses = losses_in + losses_out

            return apply_reduction(losses, self.reduction)
        else:
            raise ValueError(f"Unsupported input shape: {logits.shape}")
