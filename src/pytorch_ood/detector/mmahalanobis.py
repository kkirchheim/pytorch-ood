"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.MultiMahalanobis
    :members:
    :inherited-members:
    :show-inheritance:
"""

import logging
from typing import List, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.utils.data import DataLoader

from ..api import ModelNotSetException, RequiresFittingException, StructuredDetector
from ..utils import contains_unknown, extract_feature_avg, is_unknown

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class MultiMahalanobis(StructuredDetector):
    """
    Implements the Mahalanobis Method from the paper *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks* which supports several layers.

    For each of the given :math:`i` layers, the method calculates a class center :math:`\\mu_{iy}` for each class,
    and a shared covariance matrix :math:`\\Sigma_i` from the data.
    The per-layer outlier scores are calculated as

    .. math :: M_i(x) = - \\max_k \\lbrace (f_i(x) - \\mu_{ik})^{\\top} \\Sigma_i^{-1} (f_i(x) - \\mu_{ik}) \\rbrace

    The final outlier score is the sum of all scores, weighted by :math:`\\alpha`.
    :math:`\\alpha` defaults to uniform weighting, can be set manually via the constructor,
    or fitted via logistic regression on an ID+OOD validation set with :meth:`fit_alpha` /
    :meth:`fit_alpha_structured`, following the original paper's protocol.

    Example code is provided :doc:`here <auto_examples/detectors/mmahalanobis>`

    .. note ::
        This does not yet support ODIN preprocessing.

    :see Implementation: `GitHub <https://github.com/pokaxpoka/deep_Mahalanobis_detector>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1807.03888>`__
    """

    requires_fit = True

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

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Fit one gaussian to the features of each layer. Will average over feature maps.

        :param data_loader: dataset to fit on.
        :return:
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        zs = []

        for layer_idx in range(len(self.model)):
            # NOTE: this could be done more efficiently
            model = Sequential(*self.model[: layer_idx + 1])
            log.debug(f"Extracting for layer {layer_idx}")
            z, y = extract_feature_avg(data_loader, model, device)
            log.debug(f"Extracted {z.shape} features for {y.shape[0]} samples.")

            zs.append(z)

        return self.fit_structured(zs, y)

    def fit_structured(self: Self, zs: List[Tensor], y: Tensor) -> Self:
        """
        Fit parameters of the multi variate gaussians.

        :param zs: list of features for each layer
        :param y: class labels
        :return:
        """
        device = self.device or zs[0].device

        y = y.to(device)

        # reset any previously fitted parameters so re-fitting replaces them
        # instead of appending (e.g. when re-fitting during hyperparameter search)
        self.mu = []
        self.cov = []
        self.precision = []

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

    def fit_alpha(self: Self, data_loader: DataLoader) -> Self:
        """
        Fits the per-layer weighting factors :math:`\\alpha` via logistic regression, as
        described in the paper: given a validation set containing both ID and OOD samples
        (OOD labeled ``-1``), learns the linear combination of per-layer Mahalanobis scores
        that best separates ID from OOD. Requires :meth:`fit` (or :meth:`fit_structured`) to
        have been called first, since the Gaussians are reused as-is.

        :param data_loader: validation dataset containing both ID and OOD samples.
        :return:
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        # unlike fit()/extract_feature_avg, OOD samples must be kept here
        zs = [[] for _ in self.model]
        ys = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                ys.append(y.to(device))

                for layer_idx in range(len(self.model)):
                    # NOTE: this could be done more efficiently
                    model = Sequential(*self.model[: layer_idx + 1])
                    z = model(x)
                    z = z.mean(dim=(2, 3)).view(z.shape[0], -1)
                    zs[layer_idx].append(z)

        zs = [torch.cat(z, dim=0) for z in zs]
        y = torch.cat(ys, dim=0)

        return self.fit_alpha_structured(zs, y)

    def fit_alpha_structured(self: Self, zs: List[Tensor], y: Tensor) -> Self:
        """
        Fits the per-layer weighting factors :math:`\\alpha` via logistic regression, given
        per-layer features of a validation set containing both ID and OOD samples (OOD
        labeled ``-1``). Requires :meth:`fit_structured` to have been called first, since the
        Gaussians are reused as-is.

        :param zs: list of per-layer features of the validation set
        :param y: labels of the validation set, with OOD samples marked ``-1``
        :return:
        """
        if not self.mu:
            raise RequiresFittingException

        if not contains_unknown(y):
            raise ValueError(
                "fit_alpha_structured requires a validation set containing both ID and OOD "
                "samples (OOD labeled -1); got ID-only labels."
            )

        from sklearn.linear_model import LogisticRegressionCV

        device = self.device or zs[0].device
        y = y.to(device)

        batch_size = zs[0].shape[0]
        scores = torch.empty(batch_size, len(zs), device=device)

        for layer_idx, z in enumerate(zs):
            org_device = z.device
            z = z.to(device)
            md_k = self._calc_gaussian_scores(z, layer_idx)
            z = z.to(org_device)
            scores[:, layer_idx] = -torch.max(md_k, dim=1).values

        x_np = scores.detach().cpu().numpy()
        y_np = is_unknown(y).detach().cpu().numpy().astype(int)  # 1 = OOD, 0 = ID

        # L2-regularized, with the strength picked by internal cross-validation rather than
        # fixed -- matching the official reference implementation (LogisticRegressionCV, not
        # plain LogisticRegression). Unregularized fits overfit badly here: the per-layer
        # scores are highly correlated, so an unregularized fit finds huge, opposite-signed
        # per-layer weights that fit the validation set but don't generalize to OOD test data.
        clf = LogisticRegressionCV(fit_intercept=False)
        clf.fit(x_np, y_np)

        self.alpha = clf.coef_[0].tolist()
        log.info(f"Fitted alpha={self.alpha}")
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

    @torch.no_grad()
    def predict_structured(self, zs: List[Tensor], device=None) -> Tensor:
        """
        Calculates mahalanobis distance directly on features.
        ODIN preprocessing will not be applied.

        :param zs: list of per-layer features
        :param device: device to use for computations
        """
        if not self.mu:
            raise RequiresFittingException

        if device is None:
            device = self.device or zs[0].device

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

        device = self.device or x.device
        x = x.to(device)
        zs = []

        for layer_idx in range(len(self.model)):
            # NOTE: This could be done more efficiently
            model = Sequential(*self.model[: layer_idx + 1])
            z = model(x)
            # TODO: use mean over feature planes?
            z = z.mean(dim=(2, 3)).view(z.shape[0], -1)
            zs.append(z)

        return self.predict_structured(zs, device=device)

    @property
    def n_classes(self):
        """
        Number of classes the model is fitted for
        """
        if not self.mu:
            raise RequiresFittingException

        return self.mu[0].shape[0]
