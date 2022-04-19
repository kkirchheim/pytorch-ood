"""

..  autoclass:: pytorch_ood.MCD
    :members:

"""
import torch.nn

from .api import Detector
from .softmax import Softmax


class MCD(Detector):
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

    @staticmethod
    def run(model, x, n) -> torch.Tensor:
        """
        Assumes that the model outputs logits

        :param model: neural network
        :param x: input
        :param n: number of rounds
        :return:
        """
        model.train()

        # TODO: quickfix
        dev = x.device  # list(model.parameters())[0].device

        results = None
        with torch.no_grad():
            for i in range(n):
                output = model(x).softmax(dim=1)
                if results is None:
                    results = torch.zeros(size=output.shape).to(dev)
                results += output
        results /= n
        model.eval()
        return results

    def predict(self, x: torch.Tensor, n=30) -> torch.Tensor:
        """

        .. warning:: Side effect: The module will be in evaluation mode afterwards

        :param x: input
        :param n: number of Monte Carlo Samples
        :return: averaged output of the model
        """
        return Softmax.score(MCD.run(self.model, x, n))
