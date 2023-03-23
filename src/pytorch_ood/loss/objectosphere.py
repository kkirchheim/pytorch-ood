import logging
from typing import Optional

import torch
from torch import Tensor, nn

from ..utils import apply_reduction, contains_known, contains_unknown, is_known, is_unknown
from . import EntropicOpenSetLoss

log = logging.getLogger(__name__)


class ObjectosphereLoss(nn.Module):
    """
    From the paper *Reducing Network Agnostophobia*.

    .. math::
       \\mathcal{L}(x, y) = \\mathcal{L}_E(x,y)  + \\alpha
       \\Biggl \\lbrace
       {
       \\max \\lbrace 0, \\xi - \\lVert F(x) \\rVert \\rbrace^2 \\quad \\text{if } y \\geq 0
        \\atop
       \\lVert F(x) \\rVert_2^2 \\hspace{3.7cm}  \\text{ otherwise }
       }

    where :math:`F(x)` are deep features in some layer of the model, and
    :math:`\\mathcal{L}_E` is the Entropic Open-Set Loss.


    :see Paper:
        `NeurIPS <https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf>`__

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

    def forward(self, logits: Tensor, features: Tensor, target: Tensor) -> Tensor:
        """

        :param logits: class logits :math:`f(x)`
        :param features: deep features :math:`F(x)`
        :param target: target labels :math:`y`
        :return: the loss
        """
        entropic_loss = self.entropic(logits, target)
        losses = torch.zeros(size=(logits.shape[0],)).to(logits.device)

        if contains_known(target):
            known = is_known(target)
            # todo: this can be optimized
            losses[known] = (
                (self.xi - torch.linalg.norm(features[known], ord=2, dim=1)).relu().pow(2)
            )

        if contains_unknown(target):
            unknown = is_unknown(target)
            # todo: this can be optimized
            losses[unknown] = torch.linalg.norm(features[unknown], ord=2, dim=1).pow(2)

        loss = entropic_loss + self.alpha * losses

        return apply_reduction(loss, self.reduction)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        Outlier score used by the objectosphere loss.

        :param logits: instance logits
        :return: outlier scores
        """
        softmax_scores = -logits.softmax(dim=1).max(dim=1).values
        magn = torch.linalg.norm(logits, ord=2, dim=1)
        return softmax_scores * magn
