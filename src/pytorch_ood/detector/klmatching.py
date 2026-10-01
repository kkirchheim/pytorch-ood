"""

..  autoclass:: pytorch_ood.detector.KLMatching
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ParameterDict
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    LogitsDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)

log = logging.getLogger()


class KLMatching(LogitsDetector):
    """
    Implements KL-Matching from the paper *Scaling Out-of-Distribution Detection for Real-World Settings*.

    For each class :math:`k`, a typical posterior distribution
    :math:`d_k = \\mathbb{E}_{x \\sim \\mathcal{X}_{val}}[p(y \\vert x)]` is
    estimated as the mean posterior of the fitted data (the validation set :math:`\\mathcal{X}_{val}`) of class
    :math:`k`. In this implementation, the samples are grouped by the class labels passed to
    :meth:`fit_logits`.
    During evaluation, the KL-Divergence between the observed posterior and the typical posterior
    :math:`D_{KL}[p(y \\vert x) \\Vert d_{\\hat{y}}]` of the predicted class
    :math:`\\hat{y} = \\arg\\max_y p(y \\vert x)` is used as outlier score.
    Posteriors can only be scored for classes that were fitted.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Scaling Out-of-Distribution Detection for Real-World Settings",
            venue="ICML",
            year=2022,
            url="https://arxiv.org/abs/1911.11132",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    def __init__(self, model: Optional[Module]):
        """
        :param model: neural network, is assumed to output logits. Can be ``None`` when
            using ``fit_logits(...)`` and ``predict_logits(...)`` directly.
        """
        super(KLMatching, self).__init__()
        self.model = model
        self.dists: ParameterDict = ParameterDict()  #: Typical posteriors per class

    def fit_logits(self, logits: Tensor, labels: Tensor) -> Self:
        """
        Estimates typical distributions for each class.
        Ignores OOD samples.

        :param logits: logits of shape :math:`N \\times C`
        :param labels: class labels of shape :math:`N`
        :return: the fitted detector
        """
        device = self.device or logits.device
        logits = logits.to(device)
        labels = labels.to(device)
        probabilities = logits.softmax(dim=1)

        for label in labels.unique():
            log.debug(f"Fitting class {label}")
            d_k = probabilities[labels == label].to(device).mean(dim=0)
            self.dists[str(label.item())] = Parameter(d_k)

        return self

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits predicted by the model, shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`
        :raises ValueError: if a predicted class was not fitted
        """
        p = logits.softmax(dim=1)
        return self._score_probabilities(p)

    def _score_probabilities(self, p: Tensor) -> Tensor:
        """
        Score already-computed posterior probabilities.

        :param p: probabilities predicted by the model, shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`
        """
        device = p.device
        predictions = p.argmax(dim=1)
        scores = torch.empty(size=(p.shape[0],), device=device)

        for label in predictions.unique():
            if str(label.item()) not in self.dists:
                raise ValueError(f"Label {label.item()} not fitted.")

            dist = self.dists[str(label.item())]
            class_p = p[predictions == label]
            class_d = dist.unsqueeze(0).repeat(class_p.shape[0], 1)
            d_kl = (class_p * (class_p / class_d).log()).sum(dim=1)
            scores[predictions == label] = d_kl

        return scores

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculates KL-Divergence between predicted posteriors and typical posteriors.

        :param x: input tensor, will be passed through model
        :return: outlier scores of shape :math:`B`
        """
        if len(self.dists) == 0:
            raise RequiresFittingException("KL-Matching has to be fitted on validation data.")

        if self.model is None:
            raise ModelNotSetException

        # we move the dict with the typical posteriors to the same device as the input
        # this might be not desirable in some cases, but avoids errors
        device = x.device
        self.dists.to(device)

        logits = self.model(x)
        return self.predict_logits(logits)
