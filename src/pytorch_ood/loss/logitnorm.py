import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

from pytorch_ood.api import LossInfo, Paper, Representation, Task
from pytorch_ood.utils import is_known


def logit_norm_loss(
    logits: Tensor, target: Tensor, t: float = 1.0, reduction="mean"
) -> torch.Tensor:
    """
    Cross-entropy of the logits normalized to unit L2-norm and scaled by :math:`1/\\tau`.
    OOD samples (labels :math:`< 0`) are dropped.

    :param logits: logits as predicted by the model, shape :math:`B \\times K`
    :param target: labels of shape :math:`B`
    :param t: temperature :math:`\\tau`
    :param reduction: reduction method, one of ``mean``, ``sum`` or ``none``
    :return: the loss
    """
    known = is_known(target)
    logits = logits[known]
    target = target[known]

    norm = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
    adjusted = logits / (t * norm)
    return cross_entropy(adjusted, target, reduction=reduction)


class LogitNorm(Module):
    """
    LogitNorm from the paper *Mitigating Neural Network Overconfidence with Logit Normalization*.

    Given a model :math:`f: \\mathcal{X} \\rightarrow \\mathbb{R}^K` that maps inputs to :math:`K` logits,
    this method normalizes the logits before computing the negative log-likelihood as:

    .. math::
        \\mathcal{L}(x, y) = -\\log \\Big( \\frac{  \\exp(   \\frac{f(x)_y}{ \\tau \\lVert f(x) \\rVert_2} )}{\\sum_{i=1}^K \\exp(  \\frac{ f(x)_i}{ \\tau \\lVert f(x) \\rVert_2} ) } \\Big)

    where :math:`\\tau` is a temperature value.

    Will ignore OOD inputs.
    """

    info = LossInfo(
        paper=Paper(
            title="Mitigating Neural Network Overconfidence with Logit Normalization",
            venue="ICML",
            year=2022,
            url="https://arxiv.org/abs/2205.09310",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.LOGITS},
        supervised=False,
    )

    def __init__(self, t=1.0, reduction="mean"):
        """
        :param t: temperature :math:`\\tau`.
        :param reduction: reduction method, one of ``mean``, ``sum`` or ``none``
        """
        super().__init__()
        self.t = t
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        :param logits: logits as predicted by the model, shape :math:`B \\times K`
        :param target: labels of shape :math:`B`; labels :math:`< 0` are ignored
        :return: the loss
        """
        return logit_norm_loss(logits, target, self.t, self.reduction)
