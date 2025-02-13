"""
.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.SHE
    :members:
"""

from typing import TypeVar, Callable

import torch
from pytorch_ood.api import RequiresFittingException
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import logging
from pytorch_ood.utils import extract_features, is_known, TensorBuffer

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")

log = logging.getLogger(__name__)


class SHE(Detector):
    """
    Implements Simplified Hopfield Energy from the paper
    *Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with modern Hopfield Energy*

    For each class, SHE estimates the mean feature vector :math:`S_i` of correctly classified instances.
    For some new instances with predicted class :math:`\\hat{y}`, SHE then
    uses the inner product :math:`f(x)^{\\top} S_{\\hat{y}}` as outlier score.

    :see Paper: `OpenReview <https://openreview.net/pdf?id=KkazG4lgKL>`__
    """

    def __init__(self, backbone: Callable[[Tensor], Tensor], head: Callable[[Tensor], Tensor]):
        """
        :param backbone: feature extractor
        :param head: maps feature vectors to logits
        """
        super(SHE, self).__init__()
        self.backbone = backbone
        self.head = head
        self.patterns = None
        self.is_fitted = False

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x:  model inputs
        """
        if self.backbone is None:
            raise ModelNotSetException()

        z = self.backbone(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        :param z: features as given by the model
        """
        if self.head is None:
            raise ModelNotSetException(msg="When using predict_features(), head must not be None")

        if self.patterns is None:
            raise RequiresFittingException()

        y_hat = self.head(z).argmax(dim=1)
        self.patterns = self.patterns.to(y_hat.device)

        scores = torch.sum(torch.mul(z, self.patterns[y_hat]), dim=1)
        return -scores

    def fit(self: Self, loader: DataLoader, device: str = "cpu") -> Self:
        """
        Extracts features and calculates mean patterns.

        :param loader: data to fit
        :param device: device to use for computations. If the backbone is a nn.Module, it will be moved to this device.
        """
        if isinstance(self.backbone, nn.Module):
            log.debug(f"Moving model to {device}")
            self.backbone.to(device)

        x, y = extract_features(loader, self.backbone, device=device)
        return self.fit_features(x, y, device=device)

    @torch.no_grad()
    def _filter_correct_predictions(
        self, z: Tensor, y: Tensor, device: str = "cpu", batch_size: int = 1024
    ):
        """
        :param z: a tensor of shape (N, D) or similar
        :param y: labels of shape (N,)
        :param device: device to use for computations
        :param batch_size: how many samples we process at a time
        """
        buffer = TensorBuffer()

        for start_idx in range(0, z.size(0), batch_size):
            end_idx = start_idx + batch_size

            z_batch = z[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)

            y_hat_batch = self.head(z_batch).argmax(dim=1)

            mask = y_hat_batch == y_batch
            buffer.append("z", z_batch[mask])
            buffer.append("y", y_hat_batch[mask])

        return buffer["z"], buffer["y"]

    def fit_features(
        self: Self, z: Tensor, y: Tensor, device: str = "cpu", batch_size: int = 1024
    ) -> Self:
        """
        Calculates mean patterns per class.

        :param z: features to fit
        :param y: labels
        :param device: device to use for computations
        :param batch_size: how many samples we process at a time
        """
        if isinstance(self.backbone, nn.Module):
            log.debug(f"Moving model to {device}")
            self.backbone.to(device)

        known = is_known(y)

        if not known.any():
            raise ValueError("No ID samples")

        y = y[known]
        z = z[known]
        classes = y.unique()

        # make sure all classes are present
        assert len(classes) == classes.max().item() + 1

        z, y = self._filter_correct_predictions(z, y, device=device, batch_size=batch_size)

        m = []
        for clazz in classes:
            idx = y == clazz
            if not idx.any():
                raise ValueError(f"No correct predictions for class {clazz.item()}")

            mav = z[idx].to(device).mean(dim=0)
            m.append(mav)

        self.patterns = torch.stack(m)
        return self
