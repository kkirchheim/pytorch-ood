"""

..  autoclass:: oodtk.energy.NegativeEnergyScore
    :members:

"""
import torch


class NegativeEnergy(torch.nn.Module):
    """
    Implements the Energy Score of  *Energy-based Out-of-distribution Detection*.

    This methods calculates the negative energy for a vector of logits.
    This value can be used as outlier score.

    :param t: temperature value T

    .. math::
        E(z) = -T \\log{\\sum_i  e^{-z_i/T}}

    :see Paper:
        https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

    :see Implementation:
        https://github.com/wetliu/energy_ood

    """

    def __init__(self, t: int = 1):
        """"""
        super(NegativeEnergy, self).__init__()
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative energy

        :param x: the class logits

        :return: negative energy
        """
        return -self.t * torch.logsumexp(-x / self.t, dim=1)
