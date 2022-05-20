"""

..  autoclass:: pytorch_ood.detector.NegativeEnergy
    :members:

"""
import torch

from ..api import Detector


class NegativeEnergy(torch.nn.Module, Detector):
    """
    Implements the Energy Score of  *Energy-based Out-of-distribution Detection*.

    This methods calculates the negative energy for a vector of logits.
    This value can be used as outlier score.

    :param t: temperature value T. Default is 1.

    .. math::
        E(z) = -T \\log{\\sum_i e^{z_i/T}}

    :see Paper:
        https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

    :see Implementation:
        https://github.com/wetliu/energy_ood

    """

    def fit(self, data_loader):
        """
        Not required.
        """
        pass

    def __init__(self, model, t: int = 1):
        """"""
        super(NegativeEnergy, self).__init__()
        self.t = t
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates neative energy
        """
        z = self.model(x)
        return NegativeEnergy.score(z, self.t)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative energy

        :param x: the class logits

        :return: Energy score
        """
        return self.forward(x)

    @staticmethod
    def score(logits, t=1):
        return -t * torch.logsumexp(logits / t, dim=1)
