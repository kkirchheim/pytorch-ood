"""

..  autoclass:: oodtk.MCD
    :members:

"""
import torch.nn

from .api import Method


class MCD(Method):
    """
    From the paper *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.
    Forward-propagates the input through the model several times with activated dropout and averages the results.

    .. math:: \\hat{y} = \\frac{1}{N} \\sum_i^{N} f(x)

    :see Paper: http://proceedings.mlr.press/v48/gal16.pdf

    .. warning:: This implementations puts the model in evaluation model. This will also affect other modules, like
        BatchNorm. This is currently a workaround.
    """

    def __init__(self, model: torch.nn.Module):
        """

        :param model: the module to use for the forward pass
        """
        self.model = model

    def fit(self, data_loader):
        """
        Not required
        """
        pass

    def predict(self, x: torch.Tensor, n=30) -> torch.Tensor:
        """

        .. warning:: Side effect: The module will be in evaluation mode afterwards

        :param x: input
        :param n: number of Monte Carlo Samples
        :return: averaged output of the model
        """
        self.model.train()

        # TODO: quickfix
        dev = list(self.model.parameters())[0].device

        results = None
        with torch.no_grad():
            for i in range(n):
                output = self.model(x)
                if results is None:
                    results = torch.zeros(size=output.shape).to(dev)
                results += output
        results /= n
        self.model.eval()

        return results
