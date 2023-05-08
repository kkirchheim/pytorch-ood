"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.KLMatching
    :members:

"""
import logging
from typing import TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ParameterDict
from torch.utils.data import DataLoader

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features

log = logging.getLogger()

Self = TypeVar("Self")


class KLMatching(Detector):
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

    def __init__(self, model: Module):
        """
        :param model: neural network, is assumed to output logits.
        """
        super(KLMatching, self).__init__()
        self.model = model
        self.dists: ParameterDict = ParameterDict()  #: Typical posteriors per class

    def fit(self: Self, data_loader: DataLoader, device="cpu") -> Self:
        """
        Estimates typical distributions for each class.
        Ignores OOD samples.

        :param data_loader: validation data loader
        :param device: device which should be used for calculations
        """
        if self.model is None:
            raise ModelNotSetException

        logits, labels = extract_features(data_loader, self.model, device)
        return self.fit_features(logits, labels, device)

    def fit_features(self: Self, logits: Tensor, labels: Tensor, device="cpu") -> Self:
        """
        Estimates typical distributions for each class.
        Ignores OOD samples.

        :param logits: logits
        :param labels: class labels
        :param device: device which should be used for calculations
        """
        logits, labels = logits.to(device), labels.to(device)
        y_hat = logits.max(dim=1).indices
        probabilities = logits.softmax(dim=1)

        for label in labels.unique():
            log.debug(f"Fitting class {label}")
            d_k = probabilities[labels == label].mean(dim=0)
            self.dists[str(label.item())] = Parameter(d_k)

        return self

    def predict_features(self, p: Tensor) -> Tensor:
        """
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

        p = self.model(x).softmax(dim=1)
        return self.predict_features(p)
