import torch


def zero_grad(x):
    if type(x) is torch.Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(model: torch.nn.Module, criterion, x, y, eps=0.05, temperature=1000):
    """
    :param model: module to backprop through
    :param criterion: loss function to use. Original Implementation used NLL
    :param x: sample to preprocess
    :param y: the label the model predicted for the sample
    :param eps: size of the gradient step to take
    :param temperature: temperature to use for temperature scaling

    .. note::
        In the original implementation, the authors normalized the gradient to the image space.
        This was not described in the paper, and we have not implemented it.


    :see implementation: https://github.com/facebookresearch/odin/blob/master/code/calData.py
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