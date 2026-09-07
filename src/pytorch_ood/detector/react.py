"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightgreen?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.ReAct
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..api import FeatureMapsDetector, ModelNotSetException, RequiresFittingException
from ..utils import is_known
from .energy import EnergyBased

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class ReAct(FeatureMapsDetector):
    """
    Implements ReAct from the paper
    *ReAct: Out-of-distribution Detection With Rectified Activations*.

    ReAct clips the activations in some layer of the network (backbone) and forward propagates the
    result through the remainder of the model (head).
    In the paper, ReAct is applied to the penultimate layer of the network.

    The clipping threshold is the :math:`p`-th percentile of the activations measured on
    in-distribution training data, where :math:`p` defaults to ``90``. Call :func:`fit`
    (or :func:`fit_feature_maps`) to estimate it, or pass an explicit ``threshold``.

    The output of the network is then passed to an outlier detector that maps the output of
    the model to outlier scores.

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = ReAct(
            backbone = model.feature_maps,
            head = model.forward_feature_maps,
            detector = EnergyBased.score
        )
        detector.fit(train_loader)
        scores = detector(images)

    :see Paper: `ArXiv <https://arxiv.org/abs/2111.12797>`__
    """

    requires_fit = True

    #: Default search space for :class:`pytorch_ood.utils.GridSearch`, matching the
    #: percentile sweep used by OpenOOD (expressed here as fractions in ``[0, 1]``).
    #: Tuning re-fits the detector so the threshold is re-estimated for each percentile.
    hyperparameter_space = {"percentile": [0.85, 0.90, 0.95, 0.99]}

    def __init__(
        self,
        backbone: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        threshold: Optional[float] = None,
        percentile: float = 0.9,
        detector: Callable[[Tensor], Tensor] = None,
    ):
        """
        :param backbone: first part of model to use, should output feature maps
        :param head: second part of model used after clipping, should output logits
        :param threshold: cutoff for activations. If ``None`` (default), it is estimated
            from training data by :func:`fit`; if given, it is used directly and fitting
            is optional.
        :param percentile: fraction in :math:`(0, 1)` used as the activation percentile
            when estimating the threshold during fitting. Default ``0.9`` (the paper's value).
        :param detector: detector that maps outputs to outlier scores. Default is energy based.
        """
        self.backbone = backbone
        self.head = head
        self.percentile = percentile
        self.threshold = threshold
        self._is_fitted = threshold is not None
        self.detector = detector or EnergyBased.score

    @torch.no_grad()
    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Estimate the clipping threshold from the activations of in-distribution data.

        :param data_loader: training data; OOD samples (label ``< 0``) are ignored
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        activations = []
        for x, y in data_loader:
            known = is_known(y)
            if not known.any():
                continue
            z = self.backbone(x[known].to(device))
            activations.append(z.detach().flatten().cpu())

        if not activations:
            raise ValueError("No ID samples")

        self._set_threshold_from_activations(torch.cat(activations))
        return self

    def fit_feature_maps(self: Self, feature_maps: Tensor, y: Tensor) -> Self:
        """
        Estimate the clipping threshold directly from in-distribution feature maps.

        :param feature_maps: training feature maps
        :param y: corresponding labels; OOD samples (label ``< 0``) are ignored
        """
        known = is_known(y)
        if not known.any():
            raise ValueError("No ID samples")

        self._set_threshold_from_activations(feature_maps[known].detach().flatten().cpu())
        return self

    def _set_threshold_from_activations(self, activations: Tensor) -> None:
        self.threshold = float(np.percentile(activations.numpy(), self.percentile * 100))
        self._is_fitted = True

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through network
        """
        device = self.device
        if device is not None:
            x = x.to(device)
        x = self.backbone(x)
        return self.predict_feature_maps(x)

    @torch.no_grad()
    def predict_feature_maps(self, x: Tensor) -> Tensor:
        """
        :raise ModelNotSetException: if no head was set
        :raise RequiresFittingException: if no threshold was set or estimated
        """
        if self.head is None:
            raise ModelNotSetException

        if self.threshold is None:
            raise RequiresFittingException()

        x = x.clip(max=self.threshold)
        x = self.head(x)
        return self.detector(x)
