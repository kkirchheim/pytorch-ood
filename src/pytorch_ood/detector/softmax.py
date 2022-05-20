"""

..  autoclass:: pytorch_ood.detector.Softmax
    :members:

"""
import torch

from ..api import Detector


class Softmax(Detector, torch.nn.Module):
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

    def __init__(self, model: torch.nn.Module, t: int = 1):
        """
        :param t: temperature value T. Default is 1.
        """
        super(Softmax, self).__init__()
        self.t = t
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts on input.
        """
        return Softmax.score(self.model(x), self.t)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        return self.forward(x)

    def fit(self):
        """
        Not required

        """
        pass

    @staticmethod
    def score(logits: torch.Tensor, t=1) -> torch.Tensor:
        return -logits.div(t).softmax(dim=1).max(dim=1).values


class MaxLogit(Detector, torch.nn.Module):
    """
    Implements the Max Logit Method for OOD Detection.

    TODO
    """

    def __init__(self, model: torch.nn.Module):
        """
        :param t: temperature value T. Default is 1.
        """
        super(MaxLogit, self).__init__()
        self.model = model

    def forward(self, x) -> torch.Tensor:
        return MaxLogit.score(self.model(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        return self.forward(x)

    def fit(self):
        """
        Not required
        """
        pass

    @staticmethod
    def score(logits: torch.Tensor) -> torch.Tensor:
        """
        :param logits: logits for samples
        """
        return logits.max(dim=1).values
