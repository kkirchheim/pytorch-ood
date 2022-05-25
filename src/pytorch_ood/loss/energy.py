"""

"""
import torch
import torch.nn as nn

from ..loss import CrossEntropyLoss
from ..utils import is_known, is_unknown


class EnergyRegularizedLoss(nn.Module):
    """
    Adds a regularization term to the cross-entropy that aims to increase the energy gap between IN and OOD samples.

    The regularization term is defined as:

    .. math:: L_{\\text{energy}} = \\mathbb{E}_{(x_{in},y) \\sim \\mathcal{D}_{in}^{train}}\\max(0, E(x_{in})) - m_{in})^2 +
        \\mathbb{E}_{x_{out} \\sim \\mathcal{D}_{out}^{train}}\\max(0, m_{out} - E(x_{out}))^2

    where :math:`E(x)` is the energy of :math:`x`.

    :see Paper:
        https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

    :see Implementation:
        https://github.com/wetliu/energy_ood
    """

    def __init__(self, alpha=1, margin_in=1, margin_out=1):
        """
        :param alpha: weighting parameter
        :param margin_in:
        :param margin_out:
        """
        super(EnergyRegularizedLoss, self).__init__()
        self.m_in = margin_in
        self.m_out = margin_out
        self.nll = CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param logits: logits
        :param y: labels
        """
        regularization = self._regularization(logits, y)
        nll = self.nll(logits, y)
        return nll + self.alpha * regularization

    def _regularization(self, logits, y):
        if is_known(y).any():
            energy_in = (self._energy(logits[is_known(y)]) - self.m_in).relu().pow(2).mean()
        else:
            energy_in = 0

        if is_unknown(y).any():
            energy_out = (self._energy(self.m_out - logits[is_unknown(y)])).relu().pow(2).mean()
        else:
            energy_out = 0

        return energy_in + energy_out

    def _energy(self, logits):
        return torch.logsumexp(logits, dim=1)
