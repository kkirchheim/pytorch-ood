import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction, is_known, is_unknown


class VOSRegLoss(nn.Module):
    """
    Adds a regularization term to the cross-entropy that aims to increase the energy gap between IN and OOD samples without hyperparameter.

    The regularization term is defined as:

    .. math::
        L_{\\text{uncertainly}} = \\mathbb{E}_{(v \\sim V)} \\left[ -\\text {log}\\frac{1}{1+\\text{exp}^{-\\phi(E(v;0))}}\\right] +  \\mathbb{E}_{(x \\sim D)} \\left[ -\\text {log} \\frac{\\text{exp}^{-\\phi(E(x;0))}}{1+\\text{exp}^{-\\phi(E(x;0))}}\\right]


    where :math:`\\phi()` is a nonlinear MLP function

    :see Paper:
        `ArXiv <https://arxiv.org/pdf/2202.01197.pdf>`__

    :see Implementation:
        `GitHub <https://github.com/deeplearning-wisc/vos/>`__

    For initialisation of :math:`\\phi` and :math:`weights\\_energy`  see `GitHub <https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py#L132>`__ .

    Notice, that this loss is more effective with scheduler and low (e.g 0.001) learningrate see `GitHub <https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py#L152>`__.
    """

    def __init__(
        self, logistic_regression, weights_energy, alpha=0.1, device="cuda:0", reduction="mean"
    ):
        """
        :param logistic_regression: torch.nn.Linear(1, 2)
        :param weights_energy: torch.nn.Linear(num_classes, 1); torch.nn.init.uniform_(weights_energy.weight)
        :param alpha: weighting parameter
        """
        super(VOSRegLoss, self).__init__()
        self.logistic_regression = logistic_regression
        self.weights_energy = weights_energy
        self.alpha = alpha
        self.device = device
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param logits: logits
        :param y: labels
        """

        regularization = self._regularization(logits, y)
        loss = cross_entropy(logits, y, reduction=self.reduction)
        return apply_reduction(loss, self.reduction) + apply_reduction(
            self.alpha * regularization, self.reduction
        )

    def _regularization(self, logits, y):
        # Permutation depends on shape of logits

        if len(logits.shape) == 4:

            logits_form = logits.permute(0, 2, 3, 1)
        else:
            logits_form = logits

        # code snippet from https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py#L245
        energy_x_in = self._energy(logits_form[is_known(y)])
        energy_v_out = self._energy(logits_form[is_unknown(y)])

        input_for_lr = torch.cat((energy_x_in, energy_v_out), -1)
        if "cuda" in self.device:
            labels_for_lr = torch.cat(
                (torch.ones(len(energy_x_in)).cuda(), torch.zeros(len(energy_v_out)).cuda()), -1
            )
        else:
            labels_for_lr = torch.cat(
                (torch.ones(len(energy_x_in)), torch.zeros(len(energy_v_out))), -1
            )

        criterion = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        output1 = self.logistic_regression(input_for_lr.view(-1, 1))
        lr_reg_loss = criterion(output1, labels_for_lr.long())

        return lr_reg_loss

    def _energy(self, logits, dim=1, keepdim=False):

        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        # from https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py#L168

        m, _ = torch.max(logits, dim=dim, keepdim=True)
        value0 = logits - m
        if keepdim is False:
            m = m.squeeze(dim)
        return -(
            m
            + torch.log(
                torch.sum(
                    F.relu(self.weights_energy.weight) * torch.exp(value0),
                    dim=dim,
                    keepdim=keepdim,
                )
            )
        )
