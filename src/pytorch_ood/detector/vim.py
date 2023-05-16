"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.ViM
    :members:

"""
import logging
from typing import Callable, TypeVar

import numpy as np
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from torch import Tensor

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class ViM(Detector):
    """
    Implements Virtual Logit Matching (ViM) from the paper *ViM: Out-Of-Distribution with Virtual-logit Matching*.

    :see Paper:
        `ArXiv <https://arxiv.org/abs/2203.10807>`__
    :see Implementation:
        `GitHub <https://github.com/haoqiwang/vim/>`__

    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        d: int,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        :param model: neural network to use, is assumed to output features
        :param d: dimensionality of the principal subspace
        :param w: weights :math:`W` of the last layer of the network
        :param b: biases :math:`b` of the last layer of the network
        """
        super(ViM, self).__init__()
        self.model = model
        self.n_dim = d
        self.w = w.detach().cpu().numpy()
        self.b = b.detach().cpu().numpy()
        self.u = -np.matmul(pinv(self.w), self.b)  # new origin
        self.principal_subspace = None
        self.alpha: float = None  #: the computed :math:`\alpha` value

    def _get_logits(self, features: np.ndarray):
        """
        Calculates logits from features

        TODO: this could be done in pytorch
        """
        return np.matmul(features, self.w.T) + self.b

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model input, will be passed through neural network
        """
        if self.model is None:
            raise ModelNotSetException

        if self.principal_subspace is None or self.alpha is None:
            raise RequiresFittingException()

        with torch.no_grad():
            features = self.model(x)

        return self.predict_features(features)

    def __repr__(self):
        return f"ViM(d={self.n_dim})"

    def fit(self: Self, data_loader, device="cpu") -> Self:
        """
        Extracts features and logits, computes principle subspace and alpha. Ignores OOD samples.

        :param data_loader: dataset to fit on
        :param device: device to use
        :return:
        """
        try:
            from sklearn.covariance import EmpiricalCovariance
        except ImportError:
            raise Exception("You need to install sklearn to use ViM.")

        if self.model is None:
            raise ModelNotSetException

        features, labels = extract_features(data_loader, self.model, device)
        return self.fit_features(features, labels)

    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features as given by the model
        """
        x = x.detach().cpu().numpy()
        logits = self._get_logits(x)

        # calculate residual
        x_p_t = norm(np.matmul(x - self.u, self.principal_subspace), axis=-1)
        vlogit = x_p_t * self.alpha
        # clip for numerical stability, float32 easily overflows in logsumexp
        energy = logsumexp(np.clip(logits, -100, 100), axis=-1)
        score = -vlogit + energy
        return -Tensor(score)

    def fit_features(self: Self, features: Tensor, labels: Tensor) -> Self:
        """
        Extracts features and logits, computes principle subspace and alpha. Ignores OOD samples.

        :param features: features
        :param labels: class labels
        :return:
        """
        try:
            from sklearn.covariance import EmpiricalCovariance
        except ImportError:
            raise Exception("You need to install sklearn to use ViM.")

        features = features.cpu().numpy()

        if features.shape[1] < self.n_dim:
            n = features.shape[1] // 2
            log.warning(
                f"{features.shape[1]=} is smaller than {self.n_dim=}. Will be adjusted to {n}"
            )
            self.n_dim = n

        logits = self._get_logits(features)

        log.info("Computing principal space ...")
        # calculate eigenvectors of the covariance matrix
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

        # select largest eigenvectors to get the principal subspace
        largest_eigvals_idx = np.argsort(eig_vals * -1)[self.n_dim :]
        self.principal_subspace = np.ascontiguousarray((eigen_vectors.T[largest_eigvals_idx]).T)

        log.info("Computing alpha ...")
        # calculate residual
        x_p_t = np.matmul(features - self.u, self.principal_subspace)
        vlogits = norm(x_p_t, axis=-1)
        self.alpha = logits.max(axis=-1).mean() / vlogits.mean()
        log.info(f"{self.alpha=:.4f}")
        return self
