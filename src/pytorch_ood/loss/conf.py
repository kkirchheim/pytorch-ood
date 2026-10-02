import torch
import torch.nn as nn
import torch.nn.functional as F

from ..api import LossInfo, Paper, Representation, Task
from ..utils import drop_unknown


class ConfidenceLoss(nn.Module):
    """
    Loss proposed in *Learning Confidence for Out-of-Distribution Detection in Neural Networks*.
    The model learns to predict a confidence :math:`c` in addition to the class membership.

    The loss minimizes the negative log-likelihood for class membership prediction.

    .. math::
        \\mathcal{L}_{NLL} + \\alpha \\mathcal{L}_c = - \\sum_{i=1}^{M} \\log(p'_{i}) y_i - \\alpha \\log(c)

        \\text{where} \\quad p_i' = c \\cdot p_i + (1-c) y_i

    Here, :math:`M` is the number of classes, :math:`y` the one-hot label, :math:`p` the softmax output
    and :math:`c \\in [0,1]` the predicted confidence.
    Both terms are averaged over the batch. Samples with labels :math:`< 0` are discarded, with a warning.

    .. note::
        * We implemented clipping for numerical stability.
        * The authors additionally used ODIN preprocessing, and, during training, gave the label as a
          hint to a random half of the batch and adapted :math:`\\alpha` to a confidence budget. These
          are part of the training procedure and not implemented here.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.loss import ConfidenceLoss


        class Model(torch.nn.Module):
            # predicts class logits and a confidence in [0, 1]
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Linear(10, 16)
                self.classifier = torch.nn.Linear(16, 3)
                self.confidence = torch.nn.Linear(16, 1)

            def forward(self, x):
                z = self.features(x).relu()
                return self.classifier(z), self.confidence(z).sigmoid()


        model = Model()
        criterion = ConfidenceLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x, y = torch.randn(8, 10), torch.randint(0, 3, (8,))
        logits, confidence = model(x)
        loss = criterion(logits, confidence, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = 1 - confidence.detach().squeeze(1)  # outlier scores
    """

    info = LossInfo(
        paper=Paper(
            title="Learning Confidence for Out-of-Distribution Detection in Neural Networks",
            venue="arXiv",
            year=2018,
            url="https://arxiv.org/abs/1802.04865",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
        inputs={Representation.LOGITS, Representation.CONFIDENCE},
        supervised=False,
    )

    def __init__(self, alpha: float = 1.0, eps: float = 1e-24):
        """
        :param alpha: :math:`\\alpha` used to balance terms
        :param eps: Clipping value :math:`\\epsilon` used for numerical stability
        """
        super(ConfidenceLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, confidence: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        :param logits: class logits of shape :math:`B \\times C`
        :param confidence: predicted confidence :math:`c \\in [0, 1]`, shape :math:`B \\times 1`
        :param target: labels of shape :math:`B` (not one-hot encoded); labels :math:`< 0` are discarded
        :return: scalar loss
        """
        target, logits, confidence = drop_unknown(target, logits, confidence)
        if len(target) == 0:
            return logits.sum() * 0.0 + confidence.sum() * 0.0

        target_prob_dist = F.one_hot(target, num_classes=logits.size(1))
        prediction = F.softmax(logits, dim=1)
        adjusted_prediction = prediction * confidence + (1 - confidence) * target_prob_dist
        adjusted_prediction = adjusted_prediction.clamp(self.eps, 1.0)
        # mean over the batch for both terms, as in the reference implementation
        loss_nll = -(torch.log(adjusted_prediction) * target_prob_dist).sum(dim=1).mean()
        loss_conf = -torch.log(confidence.clamp(self.eps, 1.0)).mean()
        return loss_nll + self.alpha * loss_conf
