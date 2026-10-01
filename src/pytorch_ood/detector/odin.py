"""

..  autoclass:: pytorch_ood.detector.ODIN
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit

.. autofunction:: pytorch_ood.detector.odin_preprocessing

"""

import logging
import warnings
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import functional as F
from typing_extensions import Self

from ..api import DetectorInfo, GradientDetector, ModelNotSetException, Paper, Task

log = logging.getLogger(__name__)


def zero_grad(x):
    if type(x) is Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(
    model: torch.nn.Module,
    x: Tensor,
    y: Optional[Tensor] = None,
    criterion: Optional[Callable[[Tensor], Tensor]] = None,
    eps: float = 0.05,
    temperature: float = 1000.0,
    norm_std: Optional[List[float]] = None,
):
    """
    Functional version of ODIN.

    :param model: module to backpropagate through
    :param x: batch to preprocess, of shape :math:`B \\times C \\times H \\times W`
    :param y: the label :math:`\\hat{y}` which is used to evaluate the loss. If none is given, the models
        prediction will be used
    :param criterion: loss function :math:`\\mathcal{L}` to use. If none is given, we will use negative log
            likelihood
    :param eps: step size :math:`\\epsilon` of the gradient descent step on the loss
    :param temperature: temperature :math:`T` to use for scaling
    :param norm_std: per-channel standard deviations (one per channel :math:`C`). The sign gradient of
        channel :math:`c` is divided by ``norm_std[c]``, so that ``eps`` refers to the unnormalized input scale.
    :return: perturbed inputs :math:`\\hat{x}`, of the same shape as ``x``
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


class ODIN(GradientDetector):
    """
    Implements ODIN from the paper *Enhancing The Reliability of Out-of-distribution Image Detection in Neural
    Networks*.

    ODIN is a preprocessing method for inputs that aims to increase the discriminability of
    the softmax outputs for ID and OOD data.

    The operation requires two forward and one backward pass.

    .. math::
        \\hat{x} = x - \\epsilon \\ \\text{sign}(\\nabla_x \\mathcal{L}(f(x) / T, \\hat{y}))

    where :math:`f` is ``model``, :math:`\\mathcal{L}` is ``criterion``, :math:`\\epsilon` is ``eps``,
    :math:`T` is ``temperature`` and :math:`\\hat{y}` is the predicted class of the network.
    The outlier score is the negative maximum softmax probability of :math:`\\hat{x}`.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks",
            venue="ICLR",
            year=2018,
            url="https://arxiv.org/abs/1706.02690",
            code="https://github.com/facebookresearch/odin/",
        ),
        tasks={Task.CLASSIFICATION},
    )

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`. The ``eps`` values
    #: assume inputs normalized by ``norm_std``.
    # The temperature and eps sweep matches the one used by OpenOOD.
    hyperparameter_space = {
        "temperature": [1, 10, 100, 1000],
        "eps": [0.0014, 0.0028],
    }

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
        :param norm_std: per-channel standard deviations used for preprocessing, see
            :func:`~pytorch_ood.detector.odin_preprocessing`
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
        Calculates softmax outlier scores on ODIN pre-processed inputs. Needs gradients with respect
        to the inputs, so it must not be wrapped in ``torch.no_grad``.

        :param x: input batch of shape :math:`B \\times C \\times H \\times W`
        :return: negative maximum softmax probability of the perturbed inputs, shape :math:`B`
        """
        device = self.device
        if device is not None:
            x = x.to(device)

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
