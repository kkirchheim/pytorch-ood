import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import apply_reduction, contains_known, contains_unknown, is_known, is_unknown

log = logging.getLogger(__name__)


class ObjectosphereLoss(nn.Module):
    """
    From the paper *Reducing Network Agnostophobia*.

    .. math::
       \\mathcal{L}(x, y) = \\mathcal{L}_E(x,y)  + \\alpha
       \\Biggl \\lbrace
       {
       \\max \\lbrace 0, \\xi - \\lVert f(x) \\rVert \\rbrace^2 \\quad \\text{if } y \\geq 0
        \\atop
       \\lVert f(x) \\rVert_2^2 \\quad \\quad \\quad  \\quad \\quad \\quad  \\quad  \\text{ otherwise }
       }

    where :math:`\\mathcal{L}_E` is the Entropic Open-Set Loss


    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(self, alpha: float = 1.0, xi: float = 1.0, reduction: Optional[str] = "mean"):
        """

        :param alpha: weight coefficient
        :param xi: minimum feature magnitude :math:`\\xi`
        """
        super(ObjectosphereLoss, self).__init__()
        self.alpha = alpha
        self.xi = xi
        self.entropic = EntropicOpenSetLoss(reduction=None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param logits: class logits
        :param target: target labels
        :return:
        """
        entropic_loss = self.entropic(logits, target)
        losses = torch.zeros(size=(logits.shape[0],)).to(logits.device)

        if contains_known(target):
            known = is_known(target)
            # todo: this can be optimized
            losses[known] = (
                (self.xi - torch.linalg.norm(logits[known], ord=2, dim=1)).relu().pow(2)
            )

        if contains_unknown(target):
            unknown = is_unknown(target)
            # todo: this can be optimized
            losses[unknown] = torch.linalg.norm(logits[unknown], ord=2, dim=1).pow(2)

        loss = entropic_loss + self.alpha * losses

        return apply_reduction(loss, self.reduction)

    @staticmethod
    def score(logits) -> torch.Tensor:
        """
        Outlier score used by the objectosphere loss.

        :param logits: instance logits
        :return: outlier scores
        """
        softmax_scores = -logits.softmax(dim=1).max(dim=1).values
        magn = torch.linalg.norm(logits, ord=2, dim=1)
        return softmax_scores * magn


class EntropicOpenSetLoss(nn.Module):
    """
    From *Reducing Network Agnostophobia*.


    .. math::
       \\mathcal{L}(x, y)
       =
       \\Biggl \\lbrace
       {
       -\\log \\sigma_y(f(x)) \\quad \\text{if } y \\geq 0
        \\atop
       \\frac{1}{C} \\sum_{c=1}^C \\sigma_c(f(x)) \\quad \\text{ otherwise }
       }

    where :math:`\\sigma` is the softmax function.


    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(self, reduction: Optional[str] = None):
        """
        :param reduction: reduction method.
        """
        super(EntropicOpenSetLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param logits: class logits
        :param target: target labels
        :return:
        """
        losses = torch.zeros(size=(logits.shape[0],)).to(logits.device)

        if contains_known(target):
            known = is_known(target)
            losses[known] = F.cross_entropy(logits[known], target[known], reduction="none")

        if contains_unknown(target):
            unknown = is_unknown(target)
            losses[unknown] = -logits[unknown].softmax(dim=1).log().mean(dim=1)

        return apply_reduction(losses, self.reduction)
