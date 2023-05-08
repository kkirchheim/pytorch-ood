"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.ODIN
    :members:

.. autofunction:: pytorch_ood.detector.odin_preprocessing

"""
import logging
import warnings
from typing import Callable, List, Optional, TypeVar

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import functional as F

from ..api import Detector, ModelNotSetException

log = logging.getLogger(__name__)

Self = TypeVar("Self")


def zero_grad(x):
    if type(x) is Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(
    model: torch.nn.Module,
    x: Tensor,
    y: Optional[Tensor] = None,
    criterion: Optional[Callable[[Tensor], Tensor]] = None,
    eps: float = 0.05,
    temperature: float = 1000,
    norm_std: Optional[List[float]] = None,
):
    """
    Functional version of ODIN.

    :param model: module to backpropagate through
    :param x: sample to preprocess
    :param y: the label :math:`\\hat{y}` which is used to evaluate the loss. If none is given, the models
        prediction will be used
    :param criterion: loss function :math:`\\mathcal{L}` to use. If none is given, we will use negative log
            likelihood
    :param eps: step size :math:`\\epsilon` of the gradient ascend step
    :param temperature: temperature :math:`T` to use for scaling
    :param norm_std: standard deviations used during preprocessing
    """
    if model is None:
        raise ModelNotSetException

    # does not work in inference mode, this sometimes collides with pytorch-lightning
    if torch.is_inference_mode_enabled():
        warnings.warn("ODIN not compatible with inference mode. Will be deactivated.")

    # we make this assignment here, because adding the default to the constructor messes with sphinx
    if criterion is None:
        criterion = F.nll_loss

    with torch.inference_mode(False):
        if torch.is_inference(x):
            x = x.clone()

        with torch.enable_grad():
            x = Variable(x, requires_grad=True)
            logits = model(x) / temperature
            if y is None:
                y = logits.max(dim=1).indices
            loss = criterion(logits, y)
            loss.backward()

            gradient = torch.sign(x.grad.data)

            if norm_std:
                for i, std in enumerate(norm_std):
                    gradient.index_copy_(
                        1,
                        torch.LongTensor([i]).to(gradient.device),
                        gradient.index_select(1, torch.LongTensor([i]).to(gradient.device)) / std,
                    )

            x_hat = x - eps * gradient

    return x_hat


class ODIN(Detector):
    """
    Implements ODIN from the paper *Enhancing The Reliability of Out-of-distribution Image Detection in Neural
    Networks*.

    ODIN is a preprocessing method for inputs that aims to increase the discriminability of
    the softmax outputs for IN and OOD data.

    The operation requires two forward and one backward pass.

    .. math::
        \\hat{x} = x - \\epsilon \\ \\text{sign}(\\nabla_x \\mathcal{L}(f(x) / T, \\hat{y}))

    where :math:`\\hat{y}` is the predicted class of the network.

    :see Paper: `ArXiv <https://arxiv.org/abs/1706.02690>`__
    :see Implementation: `GitHub <https://github.com/facebookresearch/odin/>`__

    """

    def __init__(
        self,
        model: Module,
        criterion: Optional[Callable[[Tensor], Tensor]] = None,
        eps: float = 0.05,
        temperature: float = 1000.0,
        norm_std: Optional[List[float]] = None,
    ):
        """
        :param model: module to backpropagate through
        :param criterion: loss function :math:`\\mathcal{L}` to use. If None is given, we will use negative log
            likelihood
        :param eps: step size :math:`\\epsilon` of the gradient descent step
        :param temperature: temperature :math:`T` to use for scaling
        :param norm_std: standard deviations used for normalization
        """
        super(ODIN, self).__init__()
        self.model = model

        # we make this assignment here, because adding the default to the constructor messes with sphinx
        if criterion is None:
            criterion = F.nll_loss

        self.criterion = criterion  #: criterion :math:`\mathcal{L}`
        self.eps = eps  #: size :math:`\epsilon` of the gradient step in the input space
        self.temperature = temperature  #: temperature value :math:`T`
        self.norm_std = norm_std

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculates softmax outlier scores on ODIN pre-processed inputs.

        :param x: input tensor
        :return: outlier scores for each sample
        """
        x_hat = odin_preprocessing(
            model=self.model,
            x=x,
            eps=self.eps,
            criterion=self.criterion,
            temperature=self.temperature,
            norm_std=self.norm_std,
        )
        # returning negative values so higher values indicate greater outlierness
        return -self.model(x_hat).softmax(dim=1).max(dim=1).values

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required
        """
        return self

    def predict_features(self, x: Tensor) -> Tensor:
        """
        Since ODIN requires backpropagating through the model, this method can not be used.

        :raise Exception:
        """
        raise Exception("You must use a model for ODIN")
