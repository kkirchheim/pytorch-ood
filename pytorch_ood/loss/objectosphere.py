import logging

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_ood import Softmax
from pytorch_ood.utils import contains_known, contains_unknown, is_known, is_unknown

log = logging.getLogger(__name__)


class ObjectosphereLoss(nn.Module):
    """
    From *Reducing Network Agnostophobia*.

    Uses mean reduction.

    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(self, lambda_=1.0, zetta=1.0):
        """

        :param lambda_: weight for the
        :param zetta: minimum feature magnitude
        """
        super(ObjectosphereLoss, self).__init__()
        self.lambda_ = lambda_
        self.zetta = zetta
        self.entropic = EntropicOpenSetLoss()

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
        return loss.mean()

    @staticmethod
    def score(logits) -> torch.Tensor:
        """
        Outlier score used by the objectosphere loss.

        :param logits: instance logits
        :return: scores
        """
        softmax_scores = Softmax.score(logits)
        magn = torch.linalg.norm(logits, ord=2, dim=1)
        return softmax_scores * magn


class EntropicOpenSetLoss(nn.Module):
    """
    From *Reducing Network Agnostophobia*.

    Uses no reduction.

    :see Paper:
        https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf

    """

    def __init__(self):
        """ """
        super(EntropicOpenSetLoss, self).__init__()

    def forward(self, logits, target) -> torch.Tensor:
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

        return losses
