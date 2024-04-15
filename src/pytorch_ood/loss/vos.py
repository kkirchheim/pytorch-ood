"""
Parts of this code are taken from
 code snippet from https://github.com/deeplearning-wisc/vos/blob/a449b03c7d6e120087007f506d949569c845b2ec/classification/CIFAR/train_virtual.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction, is_known, is_unknown


class VOSRegLoss(nn.Module):
    """
    Implements the loss function of  *VOS: Learning what you don’t know by virtual outlier synthesis*.

    Adds a regularization term to the cross-entropy that aims to increase the (weighted) energy gap between
    IN and OOD samples.

    The regularization term is defined as:

    .. math::
        L_{\\text{uncertainly}} = \\mathbb{E}_{v \\sim V} \\left[ -\\text {log}\\frac{1}{1+\\text{exp}^{-\\phi(E(v))}}
        \\right] +  \\mathbb{E}_{x \\sim D} \\left[ -\\text {log} \\frac{\\text{exp}^{-\\phi(E(x))}}{1+
        \\text{exp}^{-\\phi(E(x))}}\\right]


    where :math:`\\phi` is a possibly non-linear function and :math:`V` and :math:`D` are the distributions
    of the (virtual) outliers and the dataset respectively.


    :see Paper:
        `ArXiv <https://arxiv.org/pdf/2202.01197.pdf>`__

    :see Implementation:
        `GitHub <https://github.com/deeplearning-wisc/vos/>`__

    For initialisation of :math:`\\phi` and  the weights for weighted energy:

    .. code :: python

        phi = torch.nn.Linear(1, 2)
        weights = torch.nn.Linear(num_classes, 1))
        torch.nn.init.uniform_(weights_energy.weight)
        criterion = VOSRegLoss(phi, weights_energy)


    """

    def __init__(
        self,
        logistic_regression: torch.nn.Linear,
        weights_energy: torch.nn.Linear,
        alpha: float = 0.1,
        device: str = "cpu",
        reduction: str = "mean",
    ):
        """
        :param logistic_regression: :math:`\\phi` function. Can be for example a linear layer.
        :param weights_energy: neural network layer, with weights for the energy
        :param alpha: weighting parameter
        :param reduction: reduction method to apply, one of ``mean``, ``sum`` or ``none``
        :param device: For example ``cpu`` or ``cuda:0``
        """
        super(VOSRegLoss, self).__init__()
        self.logistic_regression = logistic_regression
        self.weights_energy: torch.nn.Linear = weights_energy  #: weights for energy
        self.alpha = alpha
        self.device = device
        self.reduction = reduction
        self.nll = cross_entropy

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param logits: logits
        :param y: labels
        """

        regularization = self._regularization(logits, y)
        loss = self.nll(logits, y, reduction=self.reduction)
        return apply_reduction(loss, self.reduction) + apply_reduction(
            self.alpha * regularization, self.reduction
        )

    def _regularization(self, logits, y):
        # Permutation depends on shape of logits

        if len(logits.shape) == 4:
            logits_form = logits.permute(0, 2, 3, 1)
        else:
            logits_form = logits

        energy_x_in = self._energy(logits_form[is_known(y)])
        energy_v_out = self._energy(logits_form[is_unknown(y)])

        input_for_lr = torch.cat((energy_x_in, energy_v_out), -1)
        labels_for_lr = torch.cat(
            (
                torch.ones(len(energy_x_in)).to(self.device),
                torch.zeros(len(energy_v_out)).to(self.device),
            ),
            -1,
        )

        output1 = self.logistic_regression(input_for_lr.view(-1, 1))
        lr_reg_loss = self.nll(output1, labels_for_lr.long(), reduction=self.reduction)

        return lr_reg_loss

    def _energy(self, logits, dim=1, keepdim=False):
        """
        Numerically stable implementation of the energy calculation
        """
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
