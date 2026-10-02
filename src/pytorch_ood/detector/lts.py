"""

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

from ..api import DetectorInfo, FeaturesDetector, ModelNotSetException, Paper, Task
from ..utils.utils import _check_fraction
from .energy import EnergyBased

log = logging.getLogger(__name__)


class LTS(FeaturesDetector):
    """
    Implements Logit Scaling (LTS) from
    *Logit Scaling for Out-of-Distribution Detection*.

    LTS computes a per-sample scaling factor from the penultimate-layer features :math:`z`, based
    on the ratio of the total activation mass to the mass in the top fraction :math:`p` of
    activations:

    .. math::
        S(z) = \\left( \\frac{\\sum_{i=1}^{D} \\max(0, z_i)}{\\sum_{j \\in \\text{top-}k} \\max(0, z_j)} \\right)^2

    where :math:`k = \\max(1, D - \\lfloor (1 - p) D \\rceil)` for :math:`D`-dimensional features.
    The logits :math:`f(x)` are multiplied by :math:`S(z)` and passed to ``detector``, which
    defaults to :class:`EnergyBased <pytorch_ood.detector.EnergyBased>`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.detector import LTS
        from pytorch_ood.model import load_model

        model = load_model("wrn-40-2/cifar10/crossentropy")
        detector = LTS(
            encoder=model.features,
            head=model.fc,
        )
        scores = detector(images)  # images: batch of inputs, scores have shape (B,)
    """

    info = DetectorInfo(
        paper=Paper(
            title="Logit Scaling for Out-of-Distribution Detection",
            venue="Machine Vision and Applications",
            year=2025,
            url="https://arxiv.org/abs/2409.01175",
            code="https://github.com/andrijazz/lts",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

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
            :math:`B \\times D`. Can be ``None`` when using ``predict_features(...)`` directly.
        :param head: maps features to logits :math:`f(x)`, e.g. the final linear layer.
        :param p: fraction :math:`p` of top activations used in the scaling factor.
        :param detector: scoring function applied to the scaled logits. Default is
            :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.p: float = _check_fraction("p", p)
        self.detector = detector or EnergyBased.score

    @staticmethod
    def scaling_factor(z: Tensor, p: float) -> Tensor:
        """
        Compute the per-sample scaling factor :math:`S(z)` from features.

        :param z: features of shape :math:`B \\times D`.
        :param p: fraction :math:`p` of top activations to use.
        :return: scaling factors of shape :math:`B`.
        :raises ValueError: if ``z`` is not two-dimensional.
        """
        if z.ndim != 2:
            raise ValueError(f"Expected features of shape (B, D), got {tuple(z.shape)}")
        # the paper does not mention the ReLU, the official code applies it; it only affects
        # features that can be negative (e.g. transformers), the head still receives z
        z = z.relu()
        s1 = z.sum(dim=1)
        n = z.shape[1]
        # as in the official code (percentile = 1 - p); the guard avoids k = 0 for small D
        k = max(1, n - round(n * (1 - p)))
        s2 = z.topk(k, dim=1).values.sum(dim=1)
        return (s1 / s2).square()

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute LTS scores from pre-extracted features.

        :param z: penultimate-layer features of shape :math:`B \\times D`.
        :return: outlier scores of shape :math:`B`.
        :raises ValueError: if ``z`` is not two-dimensional.
        """
        s = self.scaling_factor(z, self.p)
        return self.detector(self.head(z) * s.unsqueeze(1))

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, passed through the encoder and head.
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if no encoder was given.
        """
        if self.encoder is None:
            raise ModelNotSetException
        device = self.device
        if device is not None:
            x = x.to(device)
        features = self.encoder(x)
        return self.predict_features(features)
