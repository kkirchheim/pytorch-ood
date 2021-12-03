import torch
from torch.nn import functional as F


def zero_grad(x):
    if type(x) is torch.Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(
    model: torch.nn.Module, x, y, criterion=F.nll_loss, eps=0.05, temperature=1000
):
    """
    ODIN is a preprocessing method for inputs that aims to increase the discriminability of
    the softmax outputs for In- and Out-of-Distribution data.

    The operation requires two forward and one backward pass.

    .. math::
        \\hat{x} = x + \\epsilon \\nabla_x \\mathcal{L}(f(x) / T, \\hat{y})

    :param model: module to backpropagate through
    :param x: sample to preprocess
    :param y: the label :math:`\\hat{y}` the model predicted for the sample
    :param criterion: loss function :math:`\\mathcal{L}` to use. Original Implementation used NLL
    :param eps: step size :math:`\\epsilon` of the gradient ascend step
    :param temperature: temperature :math:`T` to use for scaling

    :see Implementation: https://github.com/facebookresearch/odin/

    .. warning::
        In the original implementation, the authors normalized the gradient to the input space.
        This was not described in the paper, and we have not implemented it.

    .. note::
        This operation has the side effect of zeroing out gradients.

    """
    model.apply(zero_grad)

    with torch.enable_grad():
        x.requires_grad = True
        logits = model(x) / temperature
        loss = criterion(logits, y)
        loss.backward()
        x_hat = torch.add(x, -eps, x.grad.sign())

    model.apply(zero_grad)

    return x_hat
