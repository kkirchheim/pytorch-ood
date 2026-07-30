"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.VRA
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, TypeVar

import numpy as np
import torch.nn
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ood.utils import is_known

from ..api import FeatureMapsDetector, ModelNotSetException, RequiresFittingException
from .energy import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class VRA(FeatureMapsDetector):
    """
    Implements VRA from the paper
    *Variational Rectified Activation for Out-of-Distribution Detection*.

    VRA is a two-sided version of ReAct that clips activations both above and below
    using percentile thresholds learned from In-Distribution data, then scores the result
    with an outlier detector (Energy-Based by default).

    Unlike ReAct, which only clips activations from above at a fixed threshold,
    VRA learns per-dimension lower and upper clipping bounds from the training data
    using configurable percentiles.

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = VRA(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )
        detector.fit(train_loader)
        scores = detector(images)

    :see Paper: `ArXiv <https://arxiv.org/abs/2302.11716>`__
    """

    requires_fit = True

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        detector: Callable[[Tensor], Tensor] = None,
    ):
        """
        :param backbone: first part of the model, should output feature maps
        :param head: second part of the model used after clipping, should output logits
        :param lower_percentile: lower percentile for clipping threshold (default 1.0)
        :param upper_percentile: upper percentile for clipping threshold (default 99.0)
        :param detector: detector that maps outputs to outlier scores. Default is energy based.
        """
        self.backbone = backbone
        self.head = head
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.detector = detector or EnergyBased.score

        self._lower_threshold = None
        self._upper_threshold = None

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through the network
        """
        if self.backbone is None:
            raise ModelNotSetException()
        if self._lower_threshold is None:
            raise RequiresFittingException()

        device = self.device
        if device is not None:
            x = x.to(device)
        z = self.backbone(x)
        return self.predict_feature_maps(z)

    @torch.no_grad()
    def predict_feature_maps(self, x: Tensor) -> Tensor:
        """
        :param x: features from the backbone
        """
        if self.head is None:
            raise ModelNotSetException()
        if self._lower_threshold is None:
            raise RequiresFittingException()

        x = self._clip(x)
        x = self.head(x)
        return self.detector(x)

    def fit_feature_maps(self: Self, z: Tensor, y: Tensor) -> Self:
        """
        Calculate per-dimension clipping thresholds from In-Distribution features.
        OOD inputs will be ignored.

        :param z: features
        :param y: labels
        """
        known = is_known(y)

        if not known.any():
            raise ValueError("No ID data")

        z = z[known].detach().cpu().numpy()

        self._lower_threshold = torch.from_numpy(
            np.percentile(z, self.lower_percentile, axis=0).astype(np.float32)
        )
        self._upper_threshold = torch.from_numpy(
            np.percentile(z, self.upper_percentile, axis=0).astype(np.float32)
        )

        log.info(
            f"Lower threshold range: [{self._lower_threshold.min():.2f}, {self._lower_threshold.max():.2f}]"
        )
        log.info(
            f"Upper threshold range: [{self._upper_threshold.min():.2f}, {self._upper_threshold.max():.2f}]"
        )
        return self

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extract features and calculate clipping thresholds. OOD inputs will be ignored.

        :param data_loader: data loader to extract features from
        """
        if self.backbone is None:
            raise ModelNotSetException()

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        # NOTE: unlike pytorch_ood.utils.extract_features (which flattens each sample
        # to a 1-D vector, for pooled-feature detectors), the per-dimension clipping
        # thresholds computed by fit_feature_maps require the spatial (N, C, H, W)
        # shape to be preserved, since predict_feature_maps() clips un-flattened
        # feature maps of that same shape.
        zs, ys = [], []
        with torch.no_grad():
            for x, y in data_loader:
                known = is_known(y)
                if not known.any():
                    continue
                zs.append(self.backbone(x[known].to(device)).detach().cpu())
                ys.append(y[known].cpu())

        if not zs:
            raise ValueError("No ID data")

        z = torch.cat(zs, dim=0)
        y = torch.cat(ys, dim=0)
        self.fit_feature_maps(z, y)
        return self

    def _clip(self, z: Tensor) -> Tensor:
        lower = self._lower_threshold.to(z.device)
        upper = self._upper_threshold.to(z.device)
        return z.clip(min=lower, max=upper)
