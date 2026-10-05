import logging
from typing import Optional

import torch
from torch import Tensor, nn

from ..api import LossInfo, Paper, Representation, Task
from ..utils import (
    apply_reduction,
    contains_known,
    contains_unknown,
    is_known,
    is_unknown,
)
from . import EntropicOpenSetLoss

log = logging.getLogger(__name__)


class ObjectosphereLoss(nn.Module):
    """
    From the paper *Reducing Network Agnostophobia*.
    Extends the :class:`EntropicOpenSetLoss <pytorch_ood.loss.EntropicOpenSetLoss>` with a term that pushes the
    feature magnitude of ID samples above :math:`\\xi` and the magnitude of OOD samples towards zero.

    .. math::
       \\mathcal{L}(x, y) = \\mathcal{L}_E(x,y)  + \\alpha
       \\Biggl \\lbrace
       {
       \\max \\lbrace 0, \\xi - \\lVert F(x) \\rVert \\rbrace^2 \\quad \\text{if } y \\geq 0
        \\atop
       \\lVert F(x) \\rVert_2^2 \\hspace{3.7cm}  \\text{ otherwise }
       }

    where :math:`F(x)` are deep features in some layer of the model, and
    :math:`\\mathcal{L}_E` is the Entropic Open-Set Loss.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.loss import ObjectosphereLoss

        encoder = torch.nn.Linear(10, 16)  # deep features
        classifier = torch.nn.Linear(16, 3)
        criterion = ObjectosphereLoss()
        optimizer = torch.optim.SGD([*encoder.parameters(), *classifier.parameters()], lr=0.01)

        x, y = torch.randn(8, 10), torch.tensor([0, 1, 2, 0, 1, 2, -1, -1])  # -1: outliers
        features = encoder(x)
        # forward() takes the logits and the deep features
        loss = criterion(classifier(features), features, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """

    info = LossInfo(
        paper=Paper(
            title="Reducing Network Agnostophobia",
            venue="NeurIPS",
            year=2018,
            url="https://proceedings.neurips.cc/paper/2018/file/48db71587df6c7c442e5b76cc723169a-Paper.pdf",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.LOGITS, Representation.FEATURES},
        supervised=True,
    )

    def __init__(self, alpha: float = 1.0, xi: float = 1.0, reduction: Optional[str] = "mean"):
        """
        :param alpha: weight :math:`\\alpha` of the objectosphere term
        :param xi: minimum feature magnitude :math:`\\xi`
        :param reduction: reduction method, one of ``mean``, ``sum`` or ``none``
        """
        super(ObjectosphereLoss, self).__init__()
        self.alpha = alpha
        self.xi = xi
        self.entropic = EntropicOpenSetLoss(reduction=None)
        self.reduction = reduction

    def forward(self, logits: Tensor, features: Tensor, target: Tensor) -> Tensor:
        """
        :param logits: class logits :math:`f(x)` of shape :math:`B \\times C`
        :param features: deep features :math:`F(x)` of shape :math:`B \\times D`
        :param target: target labels :math:`y` of shape :math:`B`; labels :math:`< 0` are OOD
        :return: the loss; of shape :math:`B` if the reduction is ``none``
        """
        entropic_loss = self.entropic(logits, target)
        losses = torch.zeros(size=(logits.shape[0],)).to(logits.device)

        if contains_known(target):
            known = is_known(target)
            # todo: this can be optimized
            losses[known] = (
                (self.xi - torch.linalg.norm(features[known], ord=2, dim=1)).relu().pow(2)
            )

        if contains_unknown(target):
            unknown = is_unknown(target)
            # todo: this can be optimized
            losses[unknown] = torch.linalg.norm(features[unknown], ord=2, dim=1).pow(2)

        loss = entropic_loss + self.alpha * losses

        return apply_reduction(loss, self.reduction)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        Outlier score used by the objectosphere loss.

        :param logits: instance logits of shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`.
            Computes :math:`-\\max_c \\sigma_c(f(x)) \\cdot \\lVert f(x) \\rVert_2`
        """
        softmax_scores = -logits.softmax(dim=1).max(dim=1).values
        magn = torch.linalg.norm(logits, ord=2, dim=1)
        return softmax_scores * magn
