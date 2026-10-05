"""

..  autoclass:: pytorch_ood.detector.NNGuide
    :members:

"""

import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from pytorch_ood.api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from pytorch_ood.utils import extract_features, is_known

log = logging.getLogger(__name__)


class NNGuide(FeaturesDetector):
    """
    Implements *Nearest Neighbor Guidance for Out-of-Distribution Detection*.

    Guides the energy score with the similarity to a bank of training features. The bank holds
    the normalized features :math:`\\hat{f}(z) = f(z) / \\lVert f(z) \\rVert_2` of the training samples
    :math:`z`, each scaled by its energy :math:`E(z)`. The guidance is the mean of the :math:`k`
    largest inner products between the normalized features of the input and the bank, and the
    outlier score is the guidance times the energy of the input, negated relative to the paper:

    .. math::
        - \\underbrace{\\frac{1}{k} \\sum_{z \\in \\mathcal{N}_k(x)}
        \\langle \\hat{f}(x),\\, E(z) \\, \\hat{f}(z) \\rangle}_{\\text{guidance}} \\cdot E(x)

    where :math:`E(x) = \\log \\sum_i \\exp(l_i(x))` is the energy of the logits :math:`l(x)` and
    :math:`\\mathcal{N}_k(x)` are the :math:`k` training samples with the largest inner products
    :math:`\\langle \\hat{f}(x), E(z) \\, \\hat{f}(z) \\rangle`.

    The paper builds the bank from a random subset of the training data (e.g. 1% for ImageNet),
    which you can do by fitting on a subset.

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = NNGuide(
            encoder=model.features,
            head=model.fc,
            k=10
        )
        detector.fit(train_loader)
        scores = detector(images)
    """

    info = DetectorInfo(
        paper=Paper(
            title="Nearest Neighbor Guidance for Out-of-Distribution Detection",
            venue="ICCV",
            year=2023,
            url="https://arxiv.org/abs/2309.14888",
            code="https://github.com/roomo7time/nnguide",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    requires_fit = True

    def __init__(
        self,
        encoder: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        k: int = 10,
    ):
        """
        :param encoder: neural network that extracts penultimate-layer features :math:`f`
        :param head: callable that maps features to logits :math:`l` (e.g., a linear layer)
        :param k: number :math:`k` of nearest neighbors for guidance
        """
        super(NNGuide, self).__init__()
        self.encoder = encoder
        self.head = head
        self.k = k
        self._scaled_features: Optional[Tensor] = None

    def fit(self, data_loader: DataLoader, device=None) -> Self:
        """
        Extract features from the data loader and build the energy-scaled feature bank.

        :param data_loader: data loader with ID training data
        :param device: device for feature extraction. If ``None``, the detector device is used, else the
            device of the encoder parameters, else ``cpu``.
        :return: self
        :raise ValueError: if the data contains fewer than :math:`k` ID samples
        """
        if device is None:
            device = self.device
            if device is None:
                if isinstance(self.encoder, torch.nn.Module):
                    device = next(self.encoder.parameters()).device
                else:
                    device = "cpu"
                log.warning(f"No device given. Will use '{device}'.")

        if isinstance(self.encoder, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            self.encoder.to(device)

        z, y = extract_features(model=self.encoder, data_loader=data_loader, device=device)
        return self.fit_features(z, y)

    def fit_features(self, z: Tensor, labels: Tensor, batch_size: int = 4096) -> Self:
        """
        Build the energy-scaled feature bank from pre-extracted features.

        :param z: features of shape :math:`N \\times D`
        :param labels: class labels of shape :math:`N`; OOD samples (label below zero) are ignored
        :param batch_size: number of samples processed at once; lower it to reduce peak GPU memory
        :return: self
        :raise ValueError: if fewer than :math:`k` ID samples are given
        """
        known = is_known(labels)
        if known.sum() < self.k:
            raise ValueError(f"Need at least k={self.k} ID samples, got {int(known.sum())}")

        device = self.device or z.device
        z = z[known].detach().float()

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        # chunked, so that peak GPU memory scales with batch_size, not with the size of the bank
        chunks = []
        with torch.no_grad():
            for start in range(0, z.shape[0], batch_size):
                z_batch = z[start : start + batch_size].to(device)
                energy = torch.logsumexp(self.head(z_batch), dim=1)
                chunks.append(F.normalize(z_batch, dim=1) * energy[:, None])
        self._scaled_features = torch.cat(chunks)

        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model inputs
        :return: outlier scores of shape :math:`B`
        :raise ModelNotSetException: if no encoder was given
        :raise RequiresFittingException: if the detector has not been fitted
        """
        if self.encoder is None:
            raise ModelNotSetException()
        if self._scaled_features is None:
            raise RequiresFittingException()

        with torch.no_grad():
            z = self.encoder(x)
        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor, batch_size: int = 65536) -> Tensor:
        """
        Compute the NNGuide outlier score from pre-extracted features.

        :param z: features of shape :math:`B \\times D`
        :param batch_size: number of bank entries compared at once; lower it to reduce peak memory
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector has not been fitted
        """
        if self._scaled_features is None:
            raise RequiresFittingException()

        device = self.device or z.device
        z = z.to(device)

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        energy = torch.logsumexp(self.head(z), dim=1)
        z_norm = F.normalize(z, dim=1)

        # exact search for the k largest inner products, merged over chunks of the bank
        top = z_norm.new_full((z.shape[0], 0), float("-inf"))
        bank = self._scaled_features
        for start in range(0, bank.shape[0], batch_size):
            similarities = z_norm @ bank[start : start + batch_size].to(device).T
            top = (
                torch.cat([top, similarities], dim=1)
                .topk(min(self.k, top.shape[1] + similarities.shape[1]), dim=1)
                .values
            )
        guidance = top.mean(dim=1)

        # higher guidance * energy = more in-distribution, so negate for outlier score
        return -(guidance * energy)
