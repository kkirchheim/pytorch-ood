"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.MultiMahalanobis
    :members:
"""

import logging
from typing import List, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.utils.data import DataLoader

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import contains_unknown, extract_feature_avg

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class MultiMahalanobis(Detector):
    """
    Implements the Mahalanobis Method from the paper *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks* which supports several layers.

    For each of the given :math:`i` layers, the method calculates a class center :math:`\\mu_{iy}` for each class,
    and a shared covariance matrix :math:`\\Sigma_i` from the data.
    The per-layer outlier scores are calculated as

    .. math :: M_i(x) = - \\max_k \\lbrace (f_i(x) - \\mu_{ik})^{\\top} \\Sigma_i^{-1} (f_i(x) - \\mu_{ik}) \\rbrace

    The final outlier score is the sum of all scores, weighted by :math:`\\alpha`.

    Example code is provided :doc:`here <auto_examples/detectors/mmahalanobis>`

    .. note ::
        This does not yet support ODIN preprocessing. Also, the :math:`\\alpha` values have to be determined manually.

    :see Implementation: `GitHub <https://github.com/pokaxpoka/deep_Mahalanobis_detector>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1807.03888>`__
    """

    def __init__(self, model: List[Module], alpha: List[float] = None):
        """
        :param model: the neural network layers :math:`f_1(\\cdot),...,f_n(\\cdot)`, output of one will be used as input to the next.
        :param alpha: weighting of the individual layers. Defaults to uniform weighting.
        """
        super(MultiMahalanobis, self).__init__()

        if len(model) == 0:
            raise ValueError("No modules given")

        self.model = model

        # parameters of Gaussians
        self.mu: List[Tensor] = []  #: Centers
        self.cov: List[Tensor] = []  #: Covariance Matrices
        self.precision: List[Tensor] = []  #: Precision Matrices

        if alpha is None:
            # uniform weighting by default if alpha is not given
            alpha = [1.0] * len(model)

        self.alpha = alpha  #: Per-layer weighting factors

    def fit(self: Self, data_loader: DataLoader, device: str = None) -> Self:
        """
        Fit one gaussian to the features of each layer. Will average over feature maps.

        :param data_loader: dataset to fit on.
        :param device: device to use
        :return:
        """
        if device is None:
            # use device of first layer
            device = list(self.model[0].parameters())[0].device
            log.warning(f"No device given. Will use '{device}'.")

        if isinstance(self.model, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            self.model.to(device)

        zs = []

        for layer_idx in range(len(self.model)):
            # NOTE: this could be done more efficiently
            model = Sequential(*self.model[: layer_idx + 1])
            log.debug(f"Extracting for layer {layer_idx}")
            z, y = extract_feature_avg(data_loader, model, device)
            log.debug(f"Extracted {z.shape} features for {y.shape[0]} samples.")

            zs.append(z)

        return self.fit_features(zs, y, device)

    def fit_features(self: Self, zs: List[Tensor], y: Tensor, device: str = None) -> Self:
        """
        Fit parameters of the multi variate gaussians.

        :param zs: list of features for each layer
        :param y: class labels
        :param device: device to use
        :return:
        """
        if device is None:
            device = zs[0].device
            log.warning(f"No device given. Will use '{device}'.")

        y = y.to(device)

        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        n_classes = len(classes)

        for layer_idx, z in enumerate(zs):
            org_device = z.device
            z = z.to(device)
            log.debug(
                f"Calculating mahalanobis parameters for layer {layer_idx} with {n_classes=} {z.shape=} {y.shape=}"
            )

            mu = torch.zeros(size=(n_classes, z.shape[-1]), device=device)
            cov = torch.zeros(size=(z.shape[-1], z.shape[-1]), device=device)

            for clazz in range(n_classes):
                idxs = y.eq(clazz)
                assert idxs.sum() != 0
                z_c = z[idxs]
                mu[clazz] = z_c.mean(dim=0)
                cov += (z_c - mu[clazz]).T.mm(z_c - mu[clazz])

            cov += torch.eye(cov.shape[0], device=cov.device) * 1e-6
            precision = torch.linalg.inv(cov)

            self.mu.append(mu)
            self.cov.append(cov)
            self.precision.append(precision)
            z = z.to(org_device)

        return self

    def _calc_gaussian_scores(self, z: Tensor, layer_idx) -> Tensor:
        """ """
        features = z.view(z.size(0), z.size(1), -1)
        features = torch.mean(features, 2)
        md_k = []

        # calculate per class scores
        for clazz in range(self.n_classes):
            centered_z = features.data - self.mu[layer_idx][clazz]
            term_gau = (
                -0.5
                * torch.mm(torch.mm(centered_z, self.precision[layer_idx]), centered_z.t()).diag()
            )
            md_k.append(term_gau.view(-1, 1))

        return torch.cat(md_k, 1)

    def predict_features(self, zs: List[Tensor], device=None) -> Tensor:
        """
        Calculates mahalanobis distance directly on features.
        ODIN preprocessing will not be applied.

        :param zs: list of per-layer features
        :param device: device to use for computations
        """
        if not self.mu:
            raise RequiresFittingException

        if not device:
            device = zs[0].shape

        batch_size = zs[0].shape[0]

        scores = torch.empty(batch_size, len(zs), device=device)

        for layer_idx, z in enumerate(zs):
            org_device = z.device
            z = z.to(device)
            md_k = self._calc_gaussian_scores(z, layer_idx)
            z = z.to(org_device)

            score = -torch.max(md_k, dim=1).values
            scores[:, layer_idx] = self.alpha[layer_idx] * score

        return scores.sum(dim=1)

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor
        """
        if not self.model:
            raise ModelNotSetException

        if not self.mu:
            raise RequiresFittingException

        zs = []

        device = x.device

        for layer_idx in range(len(self.model)):
            # NOTE: This could be done more efficiently
            model = Sequential(*self.model[: layer_idx + 1])
            z = model(x)
            # TODO: use mean over feature planes?
            z = z.mean(dim=(2, 3)).view(z.shape[0], -1)
            zs.append(z)

        return self.predict_features(zs, device=device)

    @property
    def n_classes(self):
        """
        Number of classes the model is fitted for
        """
        if not self.mu:
            raise RequiresFittingException

        return self.mu[0].shape[0]
