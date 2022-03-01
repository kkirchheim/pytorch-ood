"""

..  autoclass:: oodtk.softmax.Softmax
    :members:

"""
import torch


class Softmax(torch.nn.Module):
    """
    Implements the Softmax Thresholding Baseline Scoring.

    :see Paper:
        https://arxiv.org/abs/1610.02136
    :see Implementation:
        https://github.com/hendrycks/error-detection

    """

    def __init__(self, t: int = 1):
        """
        :param t: temperature value T
        """
        super(Softmax, self).__init__()
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative energy

        :param x: the class logits

        :return: softmax score
        """
        return -self.T * torch.logsumexp(-x / self.T, dim=1)
