""" """

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..api import LossInfo, Paper, Representation, Task
from ..utils import apply_reduction, evaluate_energy_logistic_loss, is_known, is_unknown


class EnergyMarginLoss(nn.Module):
    """
    Loss from the paper *Feed Two Birds with One Scone*.
    Trains a classifier with an energy-based OOD objective that introduces a margin to better handle
    covariate-shifted data.

    A logistic-regression function :math:`\\phi` is applied to the energy score
    :math:`E(x) = \\log \\sum_{i} e^{f_i(x)}` to separate ID from OOD samples.
    Samples with targets :math:`< 0` are OOD. For them, the margin :math:`\\eta` is subtracted from the
    energy score before :math:`\\phi` is applied, so that covariate-shifted data should end up in between ID
    and OOD data. With the ID samples :math:`\\mathcal{B}_{in}` and the OOD samples :math:`\\mathcal{B}_{out}`
    of a batch, the terms are

    .. math::
        \\mathcal{L}_{in} = \\frac{1}{|\\mathcal{B}_{in}|} \\sum_{x \\in \\mathcal{B}_{in}} \\sigma\\big(\\phi(E(x))\\big)
        \\qquad
        \\mathcal{L}_{out} = \\frac{1}{|\\mathcal{B}_{out}|} \\sum_{x \\in \\mathcal{B}_{out}}
        \\sigma\\big(-\\phi(E(x) - \\eta)\\big)

    where :math:`\\sigma` is the sigmoid, so :math:`\\mathcal{L}_{in}` is the (soft) fraction of ID samples that
    is flagged as OOD. With the cross-entropy :math:`\\mathcal{L}_{CE}` over the ID samples, the loss solves

    .. math::
        \\min \\mathcal{L}_{out} \\quad \\text{s.t.} \\quad \\mathcal{L}_{in} \\leq \\alpha, \\quad
        \\mathcal{L}_{CE} \\leq \\tau \\mathcal{L}_{0}

    where :math:`\\mathcal{L}_{0}` is the cross-entropy of the pre-trained model. The constrained problem is
    solved with the augmented Lagrangian method, which minimizes

    .. math::
        w \\, \\mathcal{L}_{out} + \\psi(\\mathcal{L}_{in} - \\alpha; \\lambda, \\beta)
        + \\psi(\\mathcal{L}_{CE} - \\tau \\mathcal{L}_{0}; \\lambda_2, \\beta_2)

    with the penalty function

    .. math::
        \\psi(c; \\lambda, \\beta) =
        \\begin{cases}
            \\lambda c + \\frac{\\beta}{2} c^2 & \\text{if } \\beta c + \\lambda \\geq 0 \\\\
            -\\frac{\\lambda^2}{2 \\beta} & \\text{otherwise}
        \\end{cases}

    The Lagrange multipliers :math:`\\lambda`, :math:`\\lambda_2` start at zero and, together with the penalty
    weights :math:`\\beta`, :math:`\\beta_2`, are updated by
    :meth:`update_hyperparameters <pytorch_ood.loss.EnergyMarginLoss.update_hyperparameters>`, which should be called
    periodically, for example once per epoch.

    Every batch has to contain both ID and OOD samples.

    :see Constrained formulation: `Training OOD Detectors in their Natural Habitats (WOODS, Katz-Samuels et al., ICML 2022) <https://arxiv.org/abs/2202.03299>`__
    """

    info = LossInfo(
        paper=Paper(
            title="Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection",
            venue="ICML",
            year=2023,
            url="https://arxiv.org/abs/2306.09158",
            code="https://github.com/deeplearning-wisc/scone",
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.LOGITS},
        supervised=True,
    )

    def __init__(
        self,
        full_train_loss: float,
        eta: float = 1.0,
        false_alarm_cutoff: float = 0.05,
        in_constraint_weight: float = 1.0,
        ce_tol: float = 2.0,
        ce_constraint_weight: float = 1.0,
        out_constraint_weight: float = 1.0,
        lr_lam: float = 1.0,
        penalty_mult: float = 1.5,
        constraint_tol: float = 0.0,
    ):
        """
        :param full_train_loss: average classification (cross-entropy) loss :math:`\\mathcal{L}_{0}` of the
            pre-trained model
        :param eta: margin :math:`\\eta` between ID and OOD; covariate-shifted data should reside in-between
        :param false_alarm_cutoff: maximum tolerated fraction :math:`\\alpha` of ID samples that are flagged as
            OOD, a fraction in :math:`[0, 1]`
        :param in_constraint_weight: initial penalty weight :math:`\\beta` of the ID constraint
        :param ce_tol: factor :math:`\\tau` on ``full_train_loss``; the ID cross-entropy is constrained to be at
            most :math:`\\tau \\mathcal{L}_{0}`
        :param ce_constraint_weight: initial penalty weight :math:`\\beta_2` of the cross-entropy constraint
        :param out_constraint_weight: weight :math:`w` of the OOD term of the objective
        :param lr_lam: learning rate :math:`\\rho` of the Lagrange multipliers :math:`\\lambda`,
            :math:`\\lambda_2`
        :param penalty_mult: factor :math:`\\kappa` by which a penalty weight is multiplied in
            :meth:`update_hyperparameters <pytorch_ood.loss.EnergyMarginLoss.update_hyperparameters>`
            if its constraint is violated by more than ``constraint_tol``
        :param constraint_tol: violation tolerance :math:`\\epsilon` of both constraints, see ``penalty_mult``
        """
        super(EnergyMarginLoss, self).__init__()
        self.register_buffer("full_train_loss", torch.tensor(full_train_loss).float())
        self.register_buffer("eta", torch.tensor(eta).float())
        self.register_buffer("false_alarm_cutoff", torch.tensor(false_alarm_cutoff).float())
        self.register_buffer("in_constraint_weight", torch.tensor(in_constraint_weight).float())
        self.register_buffer("lam", torch.tensor(0).float())
        self.register_buffer("lam2", torch.tensor(0).float())
        self.register_buffer("ce_tol", torch.tensor(ce_tol).float())
        self.register_buffer("ce_constraint_weight", torch.tensor(ce_constraint_weight).float())
        self.register_buffer("out_constraint_weight", torch.tensor(out_constraint_weight).float())
        self.register_buffer("lr_lam", torch.tensor(lr_lam).float())
        self.register_buffer("penalty_mult", torch.tensor(penalty_mult).float())
        self.register_buffer("constraint_tol", torch.tensor(constraint_tol).float())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logistic_regression: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates the augmented Lagrangian function, see :class:`EnergyMarginLoss <pytorch_ood.loss.EnergyMarginLoss>`.

        :param logits: logits of shape :math:`B \\times C`
        :param targets: labels of shape :math:`B`; labels :math:`< 0` are OOD
        :param logistic_regression: function :math:`\\phi` mapping the energy score of shape
            :math:`B \\times 1` to a logit of shape :math:`B \\times 1`, for example ``torch.nn.Linear(1, 1)``
        :return: scalar loss
        :raises ValueError: if ``logits`` are not two-dimensional, or if the batch does not contain both ID
            and OOD samples
        """
        if not is_known(targets).any() or not is_unknown(targets).any():
            raise ValueError("Every batch has to contain both ID and OOD samples")
        # for classification
        if len(logits.shape) == 2:
            energy_loss_in, energy_loss_out = self._sigmoid_loss(
                logits=logits, y=targets, logistic_regression=logistic_regression
            )
            loss_in = self._alm_in_distribution_constraint(energy_loss_in=energy_loss_in)
            loss_ce = F.cross_entropy(logits[is_known(targets)], targets[is_known(targets)])
            loss_ce = self._alm_cross_entropy_constraint(loss_ce=loss_ce)
        else:
            raise ValueError(f"Unsupported input shape: {logits.shape}")
        return apply_reduction(
            loss_ce + self.out_constraint_weight * energy_loss_out + loss_in,
            reduction=None,
        )

    def _sigmoid_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        logistic_regression: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # for classification
        energy_loss_in = torch.mean(
            torch.sigmoid(
                logistic_regression(
                    (torch.logsumexp(logits[is_known(y)], dim=1)).unsqueeze(1)
                ).squeeze()
            )
        )
        energy_loss_out = torch.mean(
            torch.sigmoid(
                -logistic_regression(
                    (torch.logsumexp(logits[is_unknown(y)], dim=1) - self.eta).unsqueeze(1)
                ).squeeze()
            )
        )
        return energy_loss_in, energy_loss_out

    def _alm_in_distribution_constraint(self, energy_loss_in: torch.Tensor) -> torch.Tensor:
        # for classification
        in_constraint_term = energy_loss_in - self.false_alarm_cutoff

        # penalty function
        if self.in_constraint_weight * in_constraint_term + self.lam >= 0:
            in_loss = in_constraint_term * self.lam + self.in_constraint_weight / 2 * torch.pow(
                in_constraint_term, 2
            )
        else:
            in_loss = -torch.pow(self.lam, 2) * 0.5 / self.in_constraint_weight
        return in_loss

    def _alm_cross_entropy_constraint(self, loss_ce: torch.Tensor) -> torch.Tensor:
        # for classification
        loss_ce_constraint = loss_ce - self.ce_tol * self.full_train_loss

        # penalty function
        if self.ce_constraint_weight * loss_ce_constraint + self.lam2 >= 0:
            loss_ce = loss_ce_constraint * self.lam2 + self.ce_constraint_weight / 2 * torch.pow(
                loss_ce_constraint, 2
            )
        else:
            loss_ce = -torch.pow(self.lam2, 2) * 0.5 / self.ce_constraint_weight
        return loss_ce

    def update_hyperparameters(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        train_loader_in: DataLoader,
        logistic_regression: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Update the Lagrange multipliers :math:`\\lambda`, :math:`\\lambda_2` and the penalty weights
        :math:`\\beta`, :math:`\\beta_2` of the augmented Lagrangian function, in place.
        Call this periodically, for example once per epoch after the optimization steps.

        The constraint violations :math:`c = \\mathcal{L}_{in} - \\alpha` and
        :math:`c_2 = \\mathcal{L}_{CE} - \\tau \\mathcal{L}_{0}` are evaluated on ``train_loader_in``, with
        ``model`` in evaluation mode. Each multiplier and penalty weight is then updated with its violation:

        .. math::
            \\lambda \\leftarrow
            \\begin{cases}
                \\lambda + \\rho c & \\text{if } \\beta c + \\lambda \\geq 0 \\\\
                \\lambda - \\rho \\lambda / \\beta & \\text{otherwise}
            \\end{cases}
            \\qquad
            \\beta \\leftarrow
            \\begin{cases}
                \\kappa \\beta & \\text{if } c > \\epsilon \\\\
                \\beta & \\text{otherwise}
            \\end{cases}

        .. warning:: This puts ``model`` into evaluation mode and leaves it there. Call ``model.train()``
            before you continue training.

        .. note:: The batches are moved to the device of this loss, so move it to the device of
            ``model`` with ``.to(device)`` first.

        :param model: model that maps inputs to logits
        :param train_loader_in: loader of in-distribution data, has to yield ``(input, label)`` batches
        :param logistic_regression: function :math:`\\phi`, see :meth:`forward <pytorch_ood.loss.EnergyMarginLoss.forward>`
        """

        avg_sigmoid_energy_losses, _, avg_ce_loss = evaluate_energy_logistic_loss(
            model, train_loader_in, logistic_regression, device=self.lam.device
        )

        # update lam
        in_term_constraint = avg_sigmoid_energy_losses - self.false_alarm_cutoff
        if in_term_constraint * self.in_constraint_weight + self.lam >= 0:
            self.lam += self.lr_lam * in_term_constraint
        else:
            self.lam += -self.lr_lam * self.lam / self.in_constraint_weight

        # update lam2
        ce_constraint = avg_ce_loss - self.ce_tol * self.full_train_loss
        if ce_constraint * self.ce_constraint_weight + self.lam2 >= 0:
            self.lam2 += self.lr_lam * ce_constraint
        else:
            self.lam2 += -self.lr_lam * self.lam2 / self.ce_constraint_weight

        # update in-distribution weight for alm
        if in_term_constraint > self.constraint_tol:
            self.in_constraint_weight *= self.penalty_mult

        if ce_constraint > self.constraint_tol:
            self.ce_constraint_weight *= self.penalty_mult
