"""
.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.SHE
    :members:

"""
from typing import TypeVar, Callable

import torch
from torch import Tensor

from ..api import Detector, ModelNotSetException
from pytorch_ood.utils import extract_features, is_known

Self = TypeVar("Self")


class SHE(Detector):
    """
    Implements Simplified Hopfield Energy from the paper
    *Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with modern Hopfield Energy*

    For each class, SHE estimates the mean feature vector :math:`S_i` of correctly classified instances.
    For some new instances with predicted class :math:`\\hat{y}`, SHE then
    uses the inner product :math:`f(x)^{\\top} S_{\\hat{y}}` as outlier score.

    :see Paper: `OpenReview <https://openreview.net/pdf?id=KkazG4lgKL>`__
    """

    def __init__(self, model: Callable[[Tensor], Tensor], head: Callable[[Tensor], Tensor]):
        """
        :param model: feature extractor
        :param head: maps feature vectors to logits
        """
        super(SHE, self).__init__()
        self.model = model
        self.head = head
        self.patterns = None

        self.is_fitted = False

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x:  model inputs
        """
        if self.model is None:
            raise ModelNotSetException

        z = self.model(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        :param z: features as given by the model
        """
        y_hat = self.head(z).argmax(dim=1)
        scores = torch.sum(torch.mul(z, self.patterns[y_hat]), dim=1)
        return -scores

    def fit(self: Self, loader, device="cpu") -> Self:
        """
        Not required
        """
        x, y = extract_features(loader, self.model, device=device)
        return self.fit_features(x.to(device), y.to(device))

    def fit_features(self: Self, z, y) -> Self:
        """
        Not required
        """
        known = is_known(y)

        if not known.any():
            raise ValueError("No IN samples")

        y = y[known]
        z = z[known]
        classes = y.unique()

        # assume all classes are present
        assert len(classes) == classes.max().item() + 1

        # select correctly classified
        y_hat = self.head(z).argmax(dim=1)
        z = z[y_hat == y]
        y = y[y_hat == y]

        m = []
        for clazz in classes:
            mav = z[y == clazz].mean(dim=0)
            m.append(mav)

        self.patterns = torch.stack(m)
        return self
