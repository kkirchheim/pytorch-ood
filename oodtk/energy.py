"""
Energy Based Models
--------------------------

..  autoclass: oodtk.energy.NegativeEnergyScore
    :members:

"""
import torch


class NegativeEnergyScore(torch.nn.Module):
    """
    Implements the Energy Based Out-of-Distribution Detection scoring method.
    """
    def __init__(self, t: int = 1):
        """

        :param t: temperature
        """
        super(NegativeEnergyScore, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative energy

        :return:
        """
        return - self.T * torch.logsumexp(input=x / self.T, dim=1)
