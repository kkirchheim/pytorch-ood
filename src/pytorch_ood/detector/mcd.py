"""

..  autoclass:: pytorch_ood.detector.MCD
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit

"""

import logging
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module
from typing_extensions import Self

from ..api import Detector, DetectorInfo, ModelNotSetException, Paper, Task

log = logging.getLogger(__name__)


class MCD(Detector):
    """
    From the paper *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.
    Forward-propagates the input through the model :math:`N` times with activated dropout and averages the results.

    In ``mean`` mode, the outlier score is calculated as

    .. math:: - \\max_y \\frac{1}{N} \\sum_n^{N} \\sigma_y(f_n(x))

    where :math:`\\sigma` is the softmax function. In ``var`` mode, the scores are calculated as

    .. math::  \\frac{1}{C} \\sum_y^C \\frac{1}{N} \\sum_n^N  ( \\sigma_y(f_n(x)) - \\mu_y )^2

    where :math:`C` is the number of classes and :math:`\\mu_y = \\frac{1}{N} \\sum_n^N \\sigma_y(f_n(x))` is the
    mean softmax probability of class :math:`y`. This is the mean over the
    per class variance (computed with Bessel's correction, i.e., dividing by :math:`N-1`), which was used in *Bayesian SegNet: Model Uncertainty in Deep Convolutional
    Encoder-Decoder Architectures for Scene Understanding*.

    :see Bayesian SegNet: `ArXiv <https://arxiv.org/abs/1511.02680>`__

    .. warning:: If the model is in evaluation mode, it is temporarily put into training mode so that dropout is
        active (BatchNorm layers stay in evaluation mode if ``batch_norm=True``), and it is set back to
        evaluation mode afterwards. This could also affect other modules. If the model is already in training mode,
        it is left unchanged, so BatchNorm layers also stay in training mode and update their running statistics,
        regardless of ``batch_norm``.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
            venue="ICML",
            year=2016,
            url="http://proceedings.mlr.press/v48/gal16.pdf",
            code=None,
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
    )

    def __init__(
        self,
        model: Module,
        samples: int = 30,
        mode: str = "var",
        batch_norm: bool = True,
    ):
        """

        :param model: the module to use for the forward pass. Should output logits.
        :param samples: number :math:`N` of stochastic forward passes
        :param mode: can be one of ``var`` or ``mean``. Default is ``var``.
        :param batch_norm: keep batch norm layers in evaluation mode
        """
        assert mode in ["mean", "var"]

        self.model = model
        self.n_samples = samples  #: number :math:`N` of samples
        self.mode = mode
        self.batch_norm = batch_norm

    def fit(self, data_loader, **kwargs) -> Self:
        """
        Not required
        """
        return self

    @staticmethod
    def _switch_mode(model: Module, batch_norm: bool = True) -> bool:
        """
        Puts the model into training mode, except for variants of the batch-norm layer.

        :param batch_norm: set to False if batch-norm should also be in training mode
        :return: true if model was switched, false otherwise
        """
        mode_switch = False

        if not model.training:
            log.debug("Putting model into training mode ... ")
            mode_switch = True

            model.train()

            if batch_norm:
                for mod in model.modules():
                    # reset batch norm layers.
                    # TODO: are there other layers?
                    if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        mod.train(False)

        return mode_switch

    @staticmethod
    @torch.no_grad()
    def run(model: Module, x: Tensor, samples: int, batch_norm=True) -> Tuple[Tensor, Tensor]:
        """
        :param model: neural network
        :param x: input
        :param samples: number of rounds
        :param batch_norm: keep batch norm layers in evaluation mode
        :return: tuple of the maximum mean softmax probability, shape :math:`B`, and the variance averaged
            over classes, shape :math:`B` (spatial dimensions, if any, are kept), on the device of ``x``
        :raises ModelNotSetException: if ``model`` is ``None``
        """
        if model is None:
            raise ModelNotSetException

        mode_switch = MCD._switch_mode(model, batch_norm)

        results = []

        for i in range(samples):
            output = model(x).softmax(dim=1).cpu()
            results.append(output)

        if mode_switch:
            log.debug("Putting model into eval mode ... ")
            model.eval()

        # samples x batch x classes
        y = torch.stack(results)

        mean, indices = y.mean(dim=0).max(dim=1)
        var = y.var(dim=0)
        mean_var = var.mean(dim=1)

        # if len(y.shape) == 5:
        #     # samples x batch x classes x height x width
        #     indices = indices.unsqueeze(1)
        #     var = torch.gather(var, 1, indices)
        #     var = var.squeeze(1)
        # else:
        #     var = var[torch.arange(y.shape[1]), indices]

        return mean.to(x.device), mean_var.to(x.device)

    @staticmethod
    @torch.no_grad()
    def run_mean(model: Module, x: Tensor, samples: int, batch_norm=True) -> Tensor:
        """
        Assumes that the model outputs logits. More memory efficient implementation.

        :param model: neural network
        :param x: input
        :param samples: number of rounds
        :param batch_norm: keep batch norm layers in evaluation mode
        :return: mean softmax output of the model, shape :math:`B \\times C`
            (or :math:`B \\times C \\times H \\times W` for segmentation)
        :raises ModelNotSetException: if ``model`` is ``None``
        """
        if model is None:
            raise ModelNotSetException

        mode_switch = MCD._switch_mode(model, batch_norm=batch_norm)

        dev = x.device

        results = None
        for i in range(samples):
            output = model(x).softmax(dim=1)
            if results is None:
                results = torch.zeros(size=output.shape, device=dev)
            results += output
        results /= samples

        if mode_switch:
            log.debug("Putting model into eval mode ... ")
            model.eval()

        return results

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input
        :return: outlier scores of shape :math:`B` (or :math:`B \\times H \\times W` for segmentation).
            In ``mean`` mode, this is the negative maximum mean softmax
            probability, in ``var`` mode the variance averaged over classes.
        """
        device = self.device
        if device is not None:
            x = x.to(device)

        if self.mode == "var":
            return MCD.run(self.model, x, self.n_samples, batch_norm=self.batch_norm)[1]

        return (
            -MCD.run_mean(self.model, x, self.n_samples, batch_norm=self.batch_norm)
            .max(dim=1)
            .values
        )
