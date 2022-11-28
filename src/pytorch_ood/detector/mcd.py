"""

..  autoclass:: pytorch_ood.detector.MCD
    :members:

"""
import logging
from typing import Optional, TypeVar

import torch
from torch import nn

from ..api import Detector

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class MCD(Detector):
    """
    From the paper *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.
    Forward-propagates the input through the model several times with activated dropout and averages the results.

    The outlier score is calculated as

    .. math:: - \\max_y \\frac{1}{N} \\sum_i^{N} \\sigma_y(f(x))

    where :math:`\\sigma` is the softmax function.

    :see Paper: `ICML <http://proceedings.mlr.press/v48/gal16.pdf>`__

    .. warning:: This implementations puts the model into evaluation mode (except for variants of the BatchNorm Layers).
        This could also affect other modules and is currently a workaround.

    """

    def __init__(self, model: nn.Module, samples: int = 30):
        """

        :param model: the module to use for the forward pass. Should output logits.
        :param samples: number of iterations
        """
        self.model = model
        self.n_samples = samples  #: number :math:`N` of samples

    def fit(self: Self, data_loader) -> Self:
        """
        Not required
        """
        return self

    @staticmethod
    def run(model: torch.nn.Module, x: torch.Tensor, samples: int) -> torch.Tensor:
        """
        Assumes that the model outputs logits.

        .. note :: Input tensor should be on the same device as the model.

        :param model: neural network
        :param x: input
        :param samples: number of rounds
        :return: averaged output of the model
        """

        mode_switch = False

        dev = x.device

        if not model.training:
            log.debug("Putting model into training mode ... ")
            mode_switch = True

            model.train()

            for mod in model.modules():
                # reset batch norm layers.
                # TODO: are there other layers?
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.train(False)

        results = None
        with torch.no_grad():
            for i in range(samples):
                output = model(x).softmax(dim=1)
                if results is None:
                    results = torch.zeros(size=output.shape).to(dev)
                results += output
        results /= samples

        if mode_switch:
            log.debug("Putting model into eval mode ... ")
            model.eval()

        return results

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input
        :return: negative maximum average class score of the model
        """
        return -MCD.run(self.model, x, self.n_samples).max(dim=1).values
