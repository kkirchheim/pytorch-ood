"""

..  autoclass:: oodtk.Softmax
    :members:

"""
import torch
from torch.nn import Parameter

from .api import Method


class Softmax(Method, torch.nn.Module):
    """
    Implements the Softmax Baseline for OOD detection.

    Optionally, implements temperature scaling, which divides the logits by a constant temperature :math:`T`
    before calculating the softmax.

    .. math:: \\max_y \\text{softmax}(z / T)_y

    :see Paper:
        https://arxiv.org/abs/1610.02136
    :see Implementation:
        https://github.com/hendrycks/error-detection

    """

    def __init__(self, model, t: int = 1):
        """
        :param t: temperature value T. Default is 1.
        """
        super(Softmax, self).__init__()
        self.t = Parameter(torch.Tensor([t]))
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts on input.
        """
        return -self.model(x).div(self.t).softmax(dim=1).max(dim=1).values

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def fit(self):
        """
        Not required

        """
        pass
