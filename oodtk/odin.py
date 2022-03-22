"""

..  autoclass:: oodtk.ODIN


"""
import torch
from torch.nn import functional as F

from .api import Method


def zero_grad(x):
    if type(x) is torch.Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(
    model: torch.nn.Module, x, y=None, criterion=F.nll_loss, eps=0.05, temperature=1000
):
    """
    ODIN is a preprocessing method for inputs that aims to increase the discriminability of
    the softmax outputs for In- and Out-of-Distribution data.

    The operation requires two forward and one backward pass.

    .. math::
        \\hat{x} = x + \\epsilon \\nabla_x \\mathcal{L}(f(x) / T, \\hat{y})

    :param model: module to backpropagate through
    :param x: sample to preprocess
    :param y: the label :math:`\\hat{y}` the model predicted for the sample. If none is given, the networks
    prediction will be used
    :param criterion: loss function :math:`\\mathcal{L}` to use. Original Implementation used NLL
    :param eps: step size :math:`\\epsilon` of the gradient ascend step
    :param temperature: temperature :math:`T` to use for scaling

    :see Implementation: https://github.com/facebookresearch/odin/

    .. warning::
        In the original implementation, the authors normalized the gradient to the input space, i.e. mean centering and
        scaling to unit norm.
        This preprocessing step is differentiable and should be added to the passed model.

    .. note::
        This operation has the side effect of zeroing out gradients.

    """
    model.apply(zero_grad)

    with torch.enable_grad():
        x.requires_grad = True
        logits = model(x) / temperature
        if y is None:
            y = logits.max(dim=1).indices
        loss = criterion(logits, y)
        loss.backward()
        x_hat = torch.add(x, -eps, x.grad.sign())

    model.apply(zero_grad)

    return x_hat


class ODIN(Method):
    """
    ODIN is a preprocessing method for inputs that aims to increase the discriminability of
    the softmax outputs for In- and Out-of-Distribution data.

    The operation requires two forward and one backward pass.

    .. math::
        \\hat{x} = x + \\epsilon \\nabla_x \\mathcal{L}(f(x) / T, \\hat{y})


    :see Implementation: https://github.com/facebookresearch/odin/

    .. warning::
        In the original implementation, the authors normalized the gradient to the input space, i.e. mean centering and
        scaling to unit norm.
        This preprocessing step is differentiable and should be added to the passed model.

    .. note::
        This operation has the side effect of zeroing out gradients.

    """

    def __init__(self, model, criterion=F.nll_loss, eps=0.05, temperature=1000):
        """

        :param model: module to backpropagate through
        :param criterion: loss function :math:`\\mathcal{L}` to use. Original Implementation used NLL
        :param eps: step size :math:`\\epsilon` of the gradient ascend step
        :param temperature: temperature :math:`T` to use for scaling

        """
        super(ODIN, self).__init__()
        self.model = model
        self.criterion = criterion
        self.eps = eps
        self.temperature = temperature

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates softmax outlier scores on ODIN pre-processed inputs.
        :param x:
        :return:
        """
        x_hat = odin_preprocessing(
            model=self.model,
            x=x,
            eps=self.eps,
            criterion=self.criterion,
            temperature=self.temperature,
        )
        return self.model(x_hat).softmax(dim=1).max(dim=1).values

    def fit(self):
        """
        Not required

        """
        pass
