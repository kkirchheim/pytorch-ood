"""

..  autoclass:: pytorch_ood.detector.DICE
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional

import torch.nn
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from pytorch_ood.utils import extract_features, is_known

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils.utils import _check_fraction
from .energy import EnergyBased

log = logging.getLogger(__name__)


class DICE(FeaturesDetector):
    """
    Implements DICE from the paper
    *DICE: Leveraging Sparsification for Out-of-Distribution Detection*.

    DICE sparsifies the weights of the last layer. During :meth:`fit`, the mean activation of each feature on the
    in-distribution data is multiplied with the weights of the last layer to obtain the contribution of each weight.
    The fraction ``p`` of weights with the lowest contributions is masked out. The logits of the masked head are
    mapped to outlier scores by ``detector``, which defaults to
    :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.

    If ``p`` is changed after fitting, the detector has to be fitted again.
    """

    info = DetectorInfo(
        paper=Paper(
            title="DICE: Leveraging Sparsification for Out-of-Distribution Detection",
            venue="ECCV",
            year=2022,
            url="https://arxiv.org/abs/2111.09805",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    # matches the sweep used by the DICE paper / OpenOOD (which give it in percent)
    #: Default search space for :class:`pytorch_ood.utils.GridSearch`: the sparsification
    #: fraction ``p`` of weight contributions dropped.
    hyperparameter_space = {"p": [0.1, 0.3, 0.5, 0.7, 0.9]}

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        w: torch.Tensor,
        b: torch.Tensor,
        p: float,
        detector: Callable[[Tensor], Tensor] = None,
    ):
        """
        :param encoder: feature encoder. Can be ``None`` when using
            ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param w: weights of last layer, shape :math:`C \\times D`
        :param b: bias of last layer, shape :math:`C`
        :param p: fraction in :math:`[0, 1]` of the weight contributions (mean ID activation times weight) that
            are masked out, starting from the smallest
        :param detector: maps the logits of the masked head to outlier scores. Default is
            :meth:`EnergyBased.score <pytorch_ood.detector.EnergyBased.score>`.
        """
        self.encoder = encoder
        self.weight = w.detach().cpu()
        self.bias = b.detach().cpu()
        self.p = _check_fraction("p", p)
        self.detector = detector or EnergyBased.score

        self._is_fitted = False

        self.masked_w = None
        self.threshold = None
        self.mean_activation = None

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input, will be passed through ``encoder``
        :return: outlier scores of shape :math:`B`
        """
        if self.encoder is None:
            raise ModelNotSetException()

        z = self.encoder(x)
        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        """
        if self.masked_w is None:
            raise RequiresFittingException()

        vote = x[:, None, :] * self.masked_w.to(x.device)
        output = vote.sum(2) + self.bias.to(x.device)
        score = self.detector(output)
        return score

    def fit_features(self, z: Tensor, y: Tensor) -> Self:
        """
        Calculates the masked weights. OOD Inputs will be ignored.

        :param z: features of shape :math:`N \\times D`
        :param y: labels of shape :math:`N`. Samples with labels below zero are ignored.
        :return: the fitted detector
        :raises ValueError: if ``y`` contains no in-distribution samples
        """
        known = is_known(y)

        if not known.any():
            raise ValueError("No ID data")

        device = self.device or z.device
        z = z[known].detach().to(device).float()
        weight = self.weight.detach().to(device).float()

        self.mean_activation = z.mean(dim=0)

        contrib = self.mean_activation[None, :] * weight
        self.threshold = torch.quantile(
            contrib.flatten(), torch.tensor(self.p, device=device)
        ).item()
        log.info(f"Threshold is {self.threshold:.2f}")
        self.masked_w = torch.where(contrib > self.threshold, weight, 0)
        self._is_fitted = True
        return self

    def fit(self, data_loader: DataLoader) -> Self:
        """
        :param data_loader: data loader to extract features from. OOD inputs will be ignored.
        :return: the fitted detector
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device=device)
        self.fit_features(z, y)
        return self
