"""

"""
import torch
import torch.nn as nn

from ..loss.crossentropy import cross_entropy
from ..utils import apply_reduction, is_known, is_unknown


def _energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


class EnergyRegularizedLoss(nn.Module):
    """
    Augments the cross-entropy by  a regularization term
    that aims to increase the energy gap between IN and OOD samples.
    This term is defined as

    .. math::
       \\mathcal{L}(x, y) = \\alpha
       \\Biggl \\lbrace
       {
      \\max(0, E(x)  - m_{in})^2 \\quad \\quad  \\quad  \\quad \\quad \\quad   \\text{if } y \\geq 0
        \\atop
       \\max(0, m_{out} - E(x))^2 \\quad \\quad \\quad  \\quad \\quad  \\text{ otherwise }
       }


    where :math:`E(x) = - \\log(\\sum_i e^{f_i(x)} )` is the energy of :math:`x`.

    :see Paper:
        `NeurIPS <https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf>`__

    :see Implementation: `GitHub <https://github.com/wetliu/energy_ood>`__
    """

    def __init__(
        self,
        alpha: float = 1.0,
        margin_in: float = 1.0,
        margin_out: float = 1.0,
        reduction: str = "mean",
    ):
        """
        :param alpha: weighting parameter
        :param margin_in:  margin energy :math:`m_{in}` for IN data
        :param margin_out: margin energy :math:`m_{out}` for OOD data
        :param reduction: can be one of ``none``, ``mean``, ``sum``
        """
        super(EnergyRegularizedLoss, self).__init__()
        self.m_in = margin_in
        self.m_out = margin_out
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates weighted sum of cross-entropy and the energy regularization term.

        :param logits: logits
        :param targets: labels
        """
        regularization = self._regularization(logits, targets)
        nll = cross_entropy(logits, targets, reduction="none")
        return apply_reduction(nll + self.alpha * regularization, reduction=self.reduction)

    def _regularization(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        energy = torch.zeros(logits.shape[0]).to(logits.device)

        known = is_known(y)
        if known.any():
            energy[known] = (_energy(logits[is_known(y)]) - self.m_in).relu().pow(2)

        if (~known).any():
            energy[~known] = (self.m_out - _energy(logits[is_unknown(y)])).relu().pow(2)

        return energy
