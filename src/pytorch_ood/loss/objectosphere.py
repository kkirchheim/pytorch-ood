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

    .. math:: \\mathcal{L}(x,y) = \\lVert f(x) \\rVert^2

    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(
        self, lambda_: float = 1.0, zetta: float = 1.0, reduction: Optional[str] = "mean"
    ):
        """

        :param lambda_: weight for the
        :param zetta: minimum feature magnitude
        """
        super(ObjectosphereLoss, self).__init__()
        self.lambda_ = lambda_
        self.zetta = zetta
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
            losses[known] = (
                (torch.linalg.norm(logits[known], ord=2, dim=1) - self.zetta).relu().pow(2)
            )

        if contains_unknown(target):
            unknown = is_unknown(target)
            # todo: this can be optimized
            losses[unknown] = torch.linalg.norm(logits[unknown], ord=2, dim=1).pow(2)

        loss = entropic_loss + self.lambda_ * losses

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
            losses[unknown] = torch.logsumexp(logits[unknown], dim=1)

        return apply_reduction(losses, self.reduction)
