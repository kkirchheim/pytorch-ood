"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.GMM
    :members:

"""

import logging
from typing import Callable, Optional, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class GMM(Detector):
    """
    Implements a Gaussian Mixture Model (GMM) based Out-of-Distribution Detector.

    Fits a GMM on penultimate-layer features of the training data and uses the
    negative log-likelihood as outlier score.

    Requires ``scikit-learn`` to be installed.

    :see Paper: `ArXiv <https://arxiv.org/abs/2303.09435>`__
    """

    def __init__(
        self,
        model: Optional[Callable[[Tensor], Tensor]],
        n_components: int = 5,
        covariance_type: str = "full",
        **gmm_kwargs,
    ):
        """
        :param model: neural network to use for feature extraction (can be ``None`` for feature-based interface)
        :param n_components: number of Gaussian components in the mixture
        :param covariance_type: type of covariance parameters (``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``)
        :param gmm_kwargs: additional keyword arguments passed to scikit-learn's
            :class:`~sklearn.mixture.GaussianMixture`
        """
        self.model = model
        self._n_components = n_components
        self._covariance_type = covariance_type
        self._gmm_kwargs = gmm_kwargs
        self._gmm = None

    def fit(self: Self, data_loader: DataLoader, device: str = None) -> Self:
        """
        Extract features and fit the GMM.

        :param data_loader: data loader with training data
        :param device: device to use for feature extraction
        """
        if self.model is None:
            raise ModelNotSetException()

        if device is None:
            device = next(self.model.parameters()).device
            log.warning(f"No device given. Will use '{device}'.")

        if isinstance(self.model, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            self.model.to(device)

        z, y = extract_features(data_loader, self.model, device)
        return self.fit_features(z, y)

    def fit_features(self: Self, z: Tensor, labels: Tensor) -> Self:
        """
        Fit the GMM directly on features. OOD-labeled samples are ignored.

        :param z: features
        :param labels: class labels
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError("You have to install scikit-learn to use this detector.")

        known = is_known(labels)
        if not known.any():
            raise ValueError("No ID samples found.")

        features = z[known].detach().cpu().numpy()

        self._gmm = GaussianMixture(
            n_components=self._n_components,
            covariance_type=self._covariance_type,
            **self._gmm_kwargs,
        )
        self._gmm.fit(features)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through the model
        """
        if self.model is None:
            raise ModelNotSetException()

        z = self.model(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculate outlier scores from features using the negative GMM log-likelihood.

        :param z: features
        :return: outlier scores (higher = more OOD)
        """
        if self._gmm is None:
            raise RequiresFittingException()

        features = z.detach().cpu().numpy()
        log_likelihood = self._gmm.score_samples(features)
        return -torch.tensor(log_likelihood, dtype=z.dtype)
