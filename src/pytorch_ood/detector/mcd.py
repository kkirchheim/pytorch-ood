"""

..  autoclass:: pytorch_ood.detector.MCD
    :members:

"""
import logging

import torch
from torch import nn

from ..api import Detector

log = logging.getLogger(__name__)


class MCD(Detector):
    """
    From the paper *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.
    Forward-propagates the input through the model several times with activated dropout and averages the results.

    The outlier score is calculated as

    .. math:: - \\max \\frac{1}{N} \\sum_i^{N} f(x)

    :see Paper: http://proceedings.mlr.press/v48/gal16.pdf

    .. warning:: This implementations puts the model in evaluation mode (except for variants of the BatchNorm Layers).
        This could also affect other modules and is currently a workaround.
    """

    def __init__(self, model: nn.Module):
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
    def run(model, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Assumes that the model outputs logits

        :param model: neural network
        :param x: input
        :param n: number of rounds
        :return:
        """
        mode_switch = False

        if not model.training:
            log.debug("Putting model into training mode ... ")
            mode_switch = True

            model.train()

            for mod in model.modules():
                # reset batch norm layers.
                # TODO: are there other layers?
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.train(False)

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

        if mode_switch:
            log.debug("Putting model into eval mode ... ")
            model.eval()

        return results

    def predict(self, x: torch.Tensor, n=30) -> torch.Tensor:
        """
        :param x: input
        :param n: number of Monte Carlo Samples
        :return: averaged output of the model
        """
        return -MCD.run(self.model, x, n).max(dim=1).values
