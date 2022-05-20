"""

..  autoclass:: pytorch_ood.detector.ODIN


"""
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from ..api import Detector


def zero_grad(x):
    if type(x) is torch.Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(
    model: torch.nn.Module,
    x,
    y=None,
    criterion=F.nll_loss,
    eps=0.05,
    temperature=1000,
    norm_std=None,
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
    :param norm_std: standard deviations used during preprocessing

    :see Implementation: https://github.com/facebookresearch/odin/

    """
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
    ODIN is a preprocessing method for inputs that aims to increase the discriminability of
    the softmax outputs for In- and Out-of-Distribution data.

    The operation requires two forward and one backward pass.

    .. math::
        \\hat{x} = x - \\epsilon \\ \\text{sign}(\\nabla_x \\mathcal{L}(f(x) / T, \\hat{y}))


    :see Implementation: https://github.com/facebookresearch/odin/

    """

    def __init__(self, model, criterion=F.nll_loss, eps=0.05, temperature=1000, norm_std=None):
        """

        :param model: module to backpropagate through
        :param criterion: loss function :math:`\\mathcal{L}` to use. Original Implementation used NLL
        :param eps: step size :math:`\\epsilon` of the gradient descent step
        :param temperature: temperature :math:`T` to use for scaling

        """
        super(ODIN, self).__init__()
        self.model = model
        self.criterion = criterion
        self.eps = eps
        self.temperature = temperature
        self.norm_std = norm_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

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
            norm_std=self.norm_std,
        )
        # returning negative values so higher values indicate greater outlierness
        return -self.model(x_hat).softmax(dim=1).max(dim=1).values

    def fit(self):
        """
        Not required
        """
        pass
