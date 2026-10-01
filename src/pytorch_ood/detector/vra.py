"""

..  autoclass:: pytorch_ood.detector.VRA
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional

import numpy as np
import torch.nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from pytorch_ood.utils import is_known

from ..api import (
    DetectorInfo,
    FeatureMapsDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils.utils import _check_fraction
from .energy import EnergyBased

log = logging.getLogger(__name__)


class VRA(FeatureMapsDetector):
    """
    Implements VRA from the paper
    *Variational Rectified Activation for Out-of-Distribution Detection*.

    VRA is a two-sided version of ReAct that clips activations both above and below
    using percentile thresholds learned from In-Distribution data, then scores the result
    with an outlier detector (:meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>` by default).

    Unlike :class:`ReAct <pytorch_ood.detector.ReAct>`, which only clips activations from above with a
    single threshold, VRA clips every feature-map element between its lower and upper percentile over the
    ID training data (the thresholds have shape :math:`C \\times H \\times W`).

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = VRA(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )
        detector.fit(train_loader)
        scores = detector(images)
    """

    info = DetectorInfo(
        paper=Paper(
            title="Variational Rectified Activation for Out-of-distribution Detection",
            venue="NeurIPS",
            year=2023,
            url="https://arxiv.org/abs/2302.11716",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    requires_fit = True

    #: grid explored by :class:`pytorch_ood.utils.GridSearch`, with fractions in :math:`[0, 1]`
    # This matches the sweep used by the OpenOOD reference implementation
    # (``percentile_high``/``percentile_low``, given there in percent).
    hyperparameter_space = {
        "upper_percentile": [0.85, 0.90, 0.95, 0.99],
        "lower_percentile": [0.01, 0.05, 0.10, 0.15],
    }

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        detector: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        :param backbone: first part of the model, should output feature maps of shape
            :math:`B \\times C \\times H \\times W`
        :param head: second part of the model used after clipping, should output logits
        :param lower_percentile: percentile of the ID activations used as the lower clipping
            threshold, a fraction in :math:`[0, 1]`
        :param upper_percentile: percentile of the ID activations used as the upper clipping
            threshold, a fraction in :math:`[0, 1]`
        :param detector: callable that maps logits of shape :math:`B \\times C` to outlier scores of
            shape :math:`B`. Defaults to :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.
        """
        self.backbone = backbone
        self.head = head
        self.lower_percentile = _check_fraction("lower_percentile", lower_percentile)
        self.upper_percentile = _check_fraction("upper_percentile", upper_percentile)
        if self.lower_percentile > self.upper_percentile:
            raise ValueError("lower_percentile must not exceed upper_percentile")
        self.detector = detector or EnergyBased.score

        self._lower_threshold = None
        self._upper_threshold = None

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input batch, will be passed through the backbone and head
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector was not fitted
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
    def predict_feature_maps(self, feature_maps: Tensor) -> Tensor:
        """
        :param feature_maps: feature maps from the backbone, of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector was not fitted
        """
        if self.head is None:
            raise ModelNotSetException()
        if self._lower_threshold is None:
            raise RequiresFittingException()

        x = self._clip(feature_maps)
        x = self.head(x)
        return self.detector(x)

    def fit_feature_maps(self, z: Tensor, y: Tensor) -> Self:
        """
        Calculate per-element clipping thresholds from In-Distribution feature maps.
        OOD inputs will be ignored.

        :param z: feature maps of shape :math:`N \\times C \\times H \\times W`
        :param y: labels of shape :math:`N`
        :return: self
        :raise ValueError: if there are no ID samples
        """
        known = is_known(y)

        if not known.any():
            raise ValueError("No ID data")

        z = z[known].detach().cpu().numpy()

        self._lower_threshold = torch.from_numpy(
            np.quantile(z, self.lower_percentile, axis=0).astype(np.float32)
        )
        self._upper_threshold = torch.from_numpy(
            np.quantile(z, self.upper_percentile, axis=0).astype(np.float32)
        )

        log.info(
            f"Lower threshold range: [{self._lower_threshold.min():.2f}, {self._lower_threshold.max():.2f}]"
        )
        log.info(
            f"Upper threshold range: [{self._upper_threshold.min():.2f}, {self._upper_threshold.max():.2f}]"
        )
        return self

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extract feature maps and calculate clipping thresholds. OOD inputs will be ignored.
        The loader has to yield ``(x, y)`` batches.

        .. note::
            All ID feature maps (:math:`C \\times H \\times W` each) are kept in CPU memory, which is
            memory intensive for large datasets.

        :param data_loader: data loader to extract features from
        :return: self
        :raise ValueError: if the loader contains no ID samples
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
