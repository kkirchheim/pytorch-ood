"""

"""
import torch
import torch.nn as nn

from ..loss import CrossEntropyLoss
from ..utils import is_known, is_unknown


def _energy(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits, dim=1)


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


    where :math:`E(x) = - \\log(\\sum_y e^{f(x)_y} )` is the energy of :math:`x`.

    :see Paper:
        `NeurIPS <https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf>`__

    :see Implementation: `GitHub <https://github.com/wetliu/energy_ood>`__
    """

    def __init__(self, alpha: float = 1, margin_in: float = 1, margin_out: float = 1):
        """
        :param alpha: weighting parameter
        :param margin_in:  margin energy :math:`m_{in}` for IN data
        :param margin_out: margin energy :math:`m_{out}` for OOD data
        """
        super(EnergyRegularizedLoss, self).__init__()
        self.m_in = margin_in
        self.m_out = margin_out
        self.nll = CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates weighted sum of cross-entropy and the energy regularization term.

        :param logits: logits
        :param targets: labels
        """
        regularization = self._regularization(logits, targets)
        nll = self.nll(logits, targets)
        return nll + self.alpha * regularization

    def _regularization(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if is_known(y).any():
            energy_in = (_energy(logits[is_known(y)]) - self.m_in).relu().pow(2).mean()
        else:
            energy_in = 0

        if is_unknown(y).any():
            energy_out = (_energy(self.m_out - logits[is_unknown(y)])).relu().pow(2).mean()
        else:
            energy_out = 0

        return energy_in + energy_out
