"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.LTS
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_features
"""

import logging
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module

from ..api import FeaturesDetector, ModelNotSetException
from .energy import EnergyBased

log = logging.getLogger(__name__)


class LTS(FeaturesDetector):
    """
    Implements Logit Scaling (LTS) from
    *Logit Scaling for Out-of-Distribution Detection*.

    LTS computes a per-sample temperature from the penultimate-layer features
    based on the ratio of total activation mass to the mass concentrated in the
    top :math:`p\\%` of activations. The logits are then divided by this temperature
    before computing an energy-based OOD score:

    .. math::
        T(z) = \\left( \\frac{\\sum_i z_i}{\\sum_{j \\in \\text{top-}p\\%} z_j} \\right)^2

    .. math::
        E(x) = -\\log \\sum_{c=1}^{C} e^{f_c(x) / T(z)}

    where :math:`z` are the penultimate-layer features and :math:`f_c(x)` is the
    :math:`c`-th logit. The temperature :math:`T(z)` is adaptively determined from
    the feature distribution, enabling feature-aware temperature scaling.

    This is a fully post-hoc method: no fitting or access to training data is required.
    Supports both classification (pooled features) and segmentation (spatial feature maps).

    :see Paper: `ArXiv <https://arxiv.org/abs/2409.01175>`__

    Example Code (Classification):

    .. code :: python

        model = WideResNet(num_classes=10, pretrained="cifar10-pt")
        detector = LTS(
            encoder=model.features,
            head=model.fc,
        )
        scores = detector(images)  # (batch_size,)

    Example Code (Segmentation):

    .. code :: python

        encoder = UNetBackbone(...)  # produces (B, C, H, W) features
        head = Conv1x1Head(...)       # produces (B, K, H, W) logits
        detector = LTS(encoder=encoder, head=head)
        scores = detector(images)  # (batch_size, H, W)

    """

    requires_fit = False

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        head: Module,
        p: float = 0.05,
        detector: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        :param encoder: feature extractor that produces pooled features :math:`z` of shape
            :math:`(B, D)`. Can be ``None`` when using ``predict_features(...)`` directly.
        :param head: maps features to logits, e.g. the final linear layer.
        :param p: fraction of top activations used in the temperature computation.
            Default is 0.05 (top 5%).
        :param detector: scoring function applied to the scaled logits.
            Default is the energy score.
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.p: float = p
        self.detector = detector or EnergyBased.score

    @staticmethod
    def temperature(z: Tensor, p: float) -> Tensor:
        """
        Compute the per-sample temperature from features.

        :param z: features of shape :math:`(B, D)` or :math:`(B, C, H, W)`.
        :param p: fraction of top activations to use.
        :return: temperatures of shape :math:`(B,)`.
        """
        b = z.shape[0]
        z_flat = z.reshape(b, -1)
        s1 = z_flat.sum(dim=1)
        k = max(1, round(z_flat.shape[1] * p))
        s2 = z_flat.topk(k, dim=1).values.sum(dim=1)
        return (s1 / s2).square()

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute LTS scores from pre-extracted features.

        Supports both classification (2D pooled features) and segmentation (4D spatial features).

        :param z: penultimate-layer features, either :math:`(B, D)` or :math:`(B, C, H, W)`.
        :return: outlier scores, either :math:`(B,)` or :math:`(B, H, W)`.
        """
        t = self.temperature(z, self.p)
        logits = self.head(z)

        # Handle both 2D logits (B, K) for classification
        # and 4D logits (B, K, H, W) for segmentation
        if logits.ndim == 2:
            # Classification: logits (B, K), temperature (B,)
            scaled_logits = logits / t.unsqueeze(1)
        elif logits.ndim == 4:
            # Segmentation: logits (B, K, H, W), temperature (B,)
            scaled_logits = logits / t.view(t.shape[0], 1, 1, 1)
        else:
            raise ValueError(f"Expected logits to be 2D or 4D, got shape {logits.shape}")

        return self.detector(scaled_logits)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, passed through the encoder and head.
        """
        if self.encoder is None:
            raise ModelNotSetException
        device = self.device
        if device is not None:
            x = x.to(device)
        features = self.encoder(x)
        return self.predict_features(features)
