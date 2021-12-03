import logging
import typing

import torch
import torch.nn.functional as F
from torch import nn

from oodtk import utils

log = logging.getLogger(__name__)


class OutlierExposureLoss(nn.Module):
    """
    From the paper *Deep Anomaly Detection With Outlier Exposure*

    :see Paper: https://arxiv.org/pdf/1812.04606v1.pdf
    """

    def __init__(self, n_classes, lmbda=1):
        """
        :param n_classes: number of classes
        :param lmbda: weighting factor
        """
        super(OutlierExposureLoss, self).__init__()
        self.n_classes = n_classes
        self.lambda_ = lmbda

    def forward(self, logits, target) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """

        :param logits: logit values of the model
        :param target: target label
        """
        assert logits.shape[1] == self.n_classes

        known = utils.is_known(target)

        logits = F.log_softmax(logits)

        if utils.contains_known(target):
            loss_ce = F.cross_entropy(logits[known], target[known])
        else:
            log.warning("No In-Distribution Samples")
            loss_ce = 0

        if utils.contains_unknown(target):
            unity = torch.ones(size=(logits[~known].shape[0], self.n_classes)) / self.n_classes
            unity = unity.to(logits.device)
            # TODO: use crossentropy instead of KL divergence
            loss_oe = F.kl_div(logits[~known], unity, log_target=False, reduction="sum")
        else:
            log.warning("No Outlier Samples")
            loss_oe = 0

        return loss_ce, self.lambda_ * loss_oe
