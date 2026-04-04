"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.KLMatching
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Optional, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ParameterDict
from ..api import LogitsDetector, ModelNotSetException, RequiresFittingException

log = logging.getLogger()

Self = TypeVar("Self")


class KLMatching(LogitsDetector):
    """
    Implements KL-Matching from the paper *Scaling Out-of-Distribution Detection for Real-World Settings*.

    For each class, an typical posterior distribution
    :math:`d_y = \\mathbb{E}_{x \\sim \\mathcal{X}_{val}}[p(y \\vert x)]` is
    estimated, where :math:`y` is the class with the maximum posterior  :math:`y = \\arg\\max_y p(y \\vert x)`,
    as predicted by the model. Note that the method does not require class labels for the validation set.
    During evaluation, the KL-Divergence between the observed and the typical posterior
    :math:`D_{KL}[p(y \\vert x) \\Vert d_y]` is used as outlier score.

    This method can also be applied to multi-class settings.

    :see Paper: `ArXiv <https://arxiv.org/abs/1911.11132>`__
    """

    requires_fit = True

    def __init__(self, model: Optional[Module]):
        """
        :param model: neural network, is assumed to output logits. Can be ``None`` when
            using ``fit_logits(...)`` and ``predict_logits(...)`` directly.
        """
        super(KLMatching, self).__init__()
        self.model = model
        self.dists: ParameterDict = ParameterDict()  #: Typical posteriors per class

    def fit_logits(self: Self, logits: Tensor, labels: Tensor) -> Self:
        """
        Estimates typical distributions for each class.
        Ignores OOD samples.

        :param logits: logits
        :param labels: class labels
        """
        device = self.device or logits.device

        probabilities = logits.softmax(dim=1)

        for label in labels.unique():
            log.debug(f"Fitting class {label}")
            d_k = probabilities[labels == label].to(device).mean(dim=0)
            self.dists[str(label.item())] = Parameter(d_k)

        return self

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits predicted by the model
        """
        p = logits.softmax(dim=1)
        return self._score_probabilities(p)

    def _score_probabilities(self, p: Tensor) -> Tensor:
        """
        Score already-computed posterior probabilities.

        :param p: probabilities predicted by the model
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
        :return: Outlier scores
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
