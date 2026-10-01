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

    A logistic-regression function :math:`\\phi` is applied to the free energy score
    :math:`\\log \\sum_{i} e^{f_i(x)}` (the log-sum-exp of the logits) to separate ID from OOD samples.
    Samples with targets :math:`< 0` are treated as OOD; their score is shifted by the margin :math:`\\eta`,
    so that covariate-shifted data should end up in between ID and OOD data.
    The loss minimizes the OOD term :math:`\\mathcal{L}_{out}` (the mean of
    :math:`\\sigma(-\\phi(\\cdot))` over the OOD samples) subject to two constraints:

    * the mean of :math:`\\sigma(\\phi(\\cdot))` over the ID samples, i.e. the (soft) fraction of ID samples that
      is flagged as OOD, must not exceed ``false_alarm_cutoff``, and
    * the ID cross-entropy must not exceed ``ce_tol`` times ``full_train_loss``, the loss of the pre-trained model.

    The constrained problem is solved with the augmented Lagrangian method: the Lagrange multipliers
    :math:`\\lambda` (ID constraint) and :math:`\\lambda_2` (cross-entropy constraint) start at zero and,
    together with the penalty weights, are updated by
    :meth:`update_hyperparameters <pytorch_ood.loss.EnergyMarginLoss.update_hyperparameters>`, which should be called
    periodically, for example once per epoch.

    Every batch has to contain both ID and OOD samples, otherwise the loss is not defined.
    Only classification is supported.

    :see Derivation: `ArXiv <https://arxiv.org/pdf/2202.03299>`__
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
        Constructor of EnergyMarginLoss

        :param full_train_loss: average classification (cross-entropy) loss of the pre-trained model
        :param eta: margin :math:`\\eta` between ID and OOD; covariate-shifted data should reside in-between
        :param false_alarm_cutoff: maximum tolerated fraction of ID samples that are flagged as OOD,
            a fraction in :math:`[0, 1]`
        :param in_constraint_weight: penalty weight of the ID constraint
        :param ce_tol: factor on ``full_train_loss``; the ID cross-entropy is constrained to be at most
            ``ce_tol * full_train_loss``
        :param ce_constraint_weight: penalty weight of the cross-entropy constraint
        :param out_constraint_weight: weight of the OOD term of the objective
        :param lr_lam: learning rate of the Lagrange multipliers :math:`\\lambda`, :math:`\\lambda_2`
        :param penalty_mult: factor by which a penalty weight is multiplied in
            :meth:`update_hyperparameters <pytorch_ood.loss.EnergyMarginLoss.update_hyperparameters>`
            if its constraint is violated by more than ``constraint_tol``
        :param constraint_tol: violation tolerance of both constraints, see ``penalty_mult``
        """
        super(EnergyMarginLoss, self).__init__()
        self.full_train_loss = torch.tensor(full_train_loss).float()
        self.eta = torch.tensor(eta).float()
        self.false_alarm_cutoff = torch.tensor(false_alarm_cutoff).float()
        self.in_constraint_weight = torch.tensor(in_constraint_weight).float()
        self.lam = torch.tensor(0).float()
        self.lam2 = torch.tensor(0).float()
        self.ce_tol = torch.tensor(ce_tol).float()
        self.ce_constraint_weight = torch.tensor(ce_constraint_weight).float()
        self.out_constraint_weight = torch.tensor(out_constraint_weight).float()
        self.lr_lam = torch.tensor(lr_lam).float()
        self.penalty_mult = torch.tensor(penalty_mult).float()
        self.constraint_tol = torch.tensor(constraint_tol).float()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logistic_regression: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates weighted sum of cross-entropy and the energy regularization term a.k.a classical Augmented Lagrangian function

        :param logits: logits of shape :math:`B \\times C`
        :param targets: labels of shape :math:`B`; labels :math:`< 0` are OOD
        :param logistic_regression: function :math:`\\phi` mapping the energy score of shape
            :math:`B \\times 1` to a logit of shape :math:`B \\times 1`, for example ``torch.nn.Linear(1, 1)``
        :return: scalar loss
        :raises ValueError: if ``logits`` are not two-dimensional
        """
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
        Update the Lagrange multipliers :math:`\\lambda`, :math:`\\lambda_2` and the penalty weights of the
        augmented Lagrangian function, in place.
        Call this periodically, for example once per epoch after the optimization steps.
        The constraint violations are evaluated on ``train_loader_in``; ``model`` is put into evaluation mode.

        :param model: model that maps inputs to logits
        :param train_loader_in: loader of in-distribution data, has to yield ``(input, label)`` batches
        :param logistic_regression: function :math:`\\phi`, see :meth:`forward <pytorch_ood.loss.EnergyMarginLoss.forward>`
        """

        avg_sigmoid_energy_losses, _, avg_ce_loss = evaluate_energy_logistic_loss(
            model, train_loader_in, logistic_regression
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
