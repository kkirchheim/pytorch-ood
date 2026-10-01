""" """

import torch
import torch.nn as nn

from ..api import LossInfo, Paper, Representation, Task
from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction, is_known, is_unknown


def _energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


class EnergyRegularizedLoss(nn.Module):
    """
    Augments the cross-entropy by a regularization term that aims to increase the energy
    gap between ID and OOD samples:

    .. math::
       \\mathcal{L} = \\mathbb{E}_{x \\sim \\mathcal{D}_{in}} \\left[ \\mathcal{L}_{CE}(x, y) \\right]
       + \\alpha \\Bigl(
       \\mathbb{E}_{x \\sim \\mathcal{D}_{in}} \\left[ \\max(0, E(x) - m_{in})^2 \\right]
       + \\mathbb{E}_{x \\sim \\mathcal{D}_{out}} \\left[ \\max(0, m_{out} - E(x))^2 \\right]
       \\Bigr)

    where :math:`E(x) = - \\log(\\sum_i e^{f_i(x)} )` is the energy of :math:`x`, and samples
    with targets :math:`< 0` are OOD. The expectations are means over the ID and over the OOD
    samples of the batch (``reduction="mean"``). For segmentation, every pixel is a sample.
    """

    info = LossInfo(
        paper=Paper(
            title="Energy-based Out-of-distribution Detection",
            venue="NeurIPS",
            year=2020,
            url="https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf",
            code="https://github.com/weitliu/energy_ood",
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
        inputs={Representation.LOGITS},
        supervised=True,
    )

    # defaults: the values the paper uses for CIFAR (Sec. 4.1)
    def __init__(
        self,
        alpha: float = 0.1,
        margin_in: float = -25.0,
        margin_out: float = -7.0,
        reduction: str = "mean",
    ):
        """
        :param alpha: weighting parameter
        :param margin_in:  margin energy :math:`m_{in}` for ID data
        :param margin_out: margin energy :math:`m_{out}` for OOD data
        :param reduction: ``mean`` gives the loss above; ``none`` the per-sample (per-pixel)
            terms :math:`\\mathcal{L}_{CE} + \\alpha \\max(\\dots)^2`, ``sum`` their sum
        """
        super(EnergyRegularizedLoss, self).__init__()
        self.m_in = margin_in
        self.m_out = margin_out
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates weighted sum of cross-entropy and the energy regularization term.

        :param logits: logits, shape :math:`B \times C` or :math:`B \times C \times H \times W`
        :param targets: labels, shape :math:`B` or :math:`B \times H \times W`
        """
        if logits.dim() not in (2, 4):
            raise ValueError(f"Unsupported input shape: {logits.shape}")
        energy = _energy(logits)
        known = is_known(targets)
        nll = cross_entropy(logits, targets, reduction="none")  # zero for OOD samples
        hinge = torch.where(
            known, (energy - self.m_in).relu().pow(2), (self.m_out - energy).relu().pow(2)
        )
        if self.reduction == "mean":
            # ID and OOD terms are averaged over their own samples, as in the paper; a plain
            # mean over the batch would weight them by their share of the batch
            return _mean(nll, known) + self.alpha * (_mean(hinge, known) + _mean(hinge, ~known))
        return apply_reduction(nll + self.alpha * hinge, reduction=self.reduction)


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # zero (keeping the graph) if the batch has no such samples
    return values[mask].mean() if mask.any() else values.sum() * 0
