"""

..  autoclass:: oodtk.mcd.MCD
    :members:

"""
from abc import ABC

import torch.nn

from .api import Method


class MCD(Method):
    """
    From the paper *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.
    Forward-propagates the input through the model several times with activated dropout and averages the results.

    .. math:: \\hat{y} = \\frac{1}{N} \\sum_i^{N} f(x)

    :see Paper: http://proceedings.mlr.press/v48/gal16.pdf
    """

    def __init__(self, model: torch.nn.Module):
        """

        :param model: the module to use for the forward pass
        """
        self.model = model

    def predict(self, x: torch.Tensor, n=30) -> torch.Tensor:
        """

        .. warning:: Side effect: The module will be in evaluation mode afterwards

        :param x: input
        :param n: number of Monte Carlo Samples
        :return: averaged output of the model
        """
        self.train()
        results = None

        with torch.no_grad():
            output = self.model(x)

            if results is None:
                results = torch.zeros(size=output.shape)

            results += output

        results /= n
        self.eval()
