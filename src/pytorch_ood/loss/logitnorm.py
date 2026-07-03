import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

from pytorch_ood.utils import is_known


def logit_norm_loss(
    logits: Tensor, target: Tensor, t: float = 1.0, reduction="mean"
) -> torch.Tensor:
    """
    :param logits:  logits as predicted by the model
    :param target:  labels
    :param t: temperature :math:`\\tau`
    :param reduction: reduction method, one of ``mean``, ``sum`` or ``none``
    """
    known = is_known(target)
    logits = logits[known]
    target = target[known]

    norm = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
    adjusted = logits / (t * norm)
    return cross_entropy(adjusted, target, reduction=reduction)


class LogitNorm(Module):
    """
    LogitNorm from  the paper *Mitigating Neural Network Overconfidence with Logit Normalization*.

    Given a model :math:`f: \\mathcal{X} \\rightarrow \\mathbb{R}^K` that maps inputs to :math:`K` logits,
    this method normalizes the logits before computing the negative log-likelihood as:

    .. math::
        \\mathcal{L}(x, y) = -\\log \\Big( \\frac{  \\exp(   \\frac{f(x)_y}{ \\tau \\lVert x  \\rVert} )}{\\sum_{i=1}^K \\exp(  \\frac{ f(x)_i}{ \\tau \\lVert x \\rVert} ) } \\Big)

    where :math:`\\tau` is a temperature  value.

    Will ignore  OOD inputs.

    :see Paper:
        `ICML <https://arxiv.org/abs/2205.09310>`__

    """

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
        :param logits:  logits as predicted by the model
        :param target:  labels
        """
        return logit_norm_loss(logits, target, self.t, self.reduction)
