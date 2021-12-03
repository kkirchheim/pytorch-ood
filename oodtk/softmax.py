"""

..  autoclass:: oodtk.softmax.SoftmaxThresholding
    :members:

"""
import torch


class Softmax(torch.nn.Module):
    """
    Implements the Softmax Thresholding Baseline Scoring.

    :param t: temperature value T

    """

    def __init__(self, t: int = 1):
        """"""
        super(Softmax, self).__init__()
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative energy

        :param x: the class logits

        :return: softmax score
        """
        return -self.T * torch.logsumexp(-x / self.T, dim=1)
