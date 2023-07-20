"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.RMD
    :members:
"""
import logging
import warnings
from typing import Callable, List, Optional, TypeVar

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import TensorBuffer, contains_unknown, extract_features, is_known, is_unknown
from pytorch_ood.detector.mahalanobis import Mahalanobis

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class RMD(Mahalanobis):
    """
    Implements the Relative Mahalanobis Distance (RMD) from the paper
    *A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection*.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.

    Additionally, it fits a background gaussian with mean :math:`\\mu_0` and covariance matrix
    :math:`\\Sigma_0` to all of the features and calculates outlier scores as

    .. math :: \\min_k \\lbrace d_k(f(x)) - d_0(f(x)) \\rbrace

    where :math:`d_k` is the mahalanobis score for class :math:`k` and :math:`d_0` is the
    mahalanobis score under the background gaussian.

    :see Paper: `ArXiv <https://arxiv.org/pdf/2106.09022.pdf>`__
    """

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
    ):
        """
        :param model: the Neural Network, should output features
        """
        super(RMD, self).__init__(model=model)

        self.background_mu = None
        self.background_cov = None
        self.background_precision = None

    def fit_features(self: Self, z: Tensor, y: Tensor, device: str = None) -> Self:
        """
        Fit parameters of the multi variate gaussian.

        :param z: features
        :param y: class labels
        :param device: device to use
        :return:
        """
        if device is None:
            device = z.device
            log.warning(f"No device given. Will use '{device}'.")

        z, y = z.to(device), y.to(device)

        super(RMD, self).fit_features(z, y, device)

        log.debug("Fitting background gaussian.")
        self.background_mu = z.mean(dim=0)
        self.background_cov = (z - self.background_mu).T.mm(z - self.background_mu)
        self.background_cov += torch.eye(self.background_cov.shape[0], device=self.background_cov.device) * 1e-6

        self.background_precision = torch.linalg.inv(self.background_cov)
        return self

    def _background_score(self, z: Tensor) -> Tensor:
        centered_z = z - self.background_mu
        return torch.mm(torch.mm(centered_z, self.background_precision), centered_z.t()).diag()

    def _class_score(self, z, k):
        centered_z = z - self.mu[k]
        return torch.mm(torch.mm(centered_z, self.precision), centered_z.t()).diag()

    def _calc_gaussian_scores(self, z: Tensor) -> Tensor:
        """
        This is written a bit differently compared to the mahalanobis paper
        """
        # mean over feature maps
        features = z.view(z.size(0), z.size(1), -1)
        features = torch.mean(features, 2)
        md_k = []

        for clazz in range(self.n_classes):
            score = self._class_score(features, clazz)
            md_k.append(score.view(-1, 1))

        return torch.cat(md_k, 1)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculates mahalanobis distance directly on features.

        :param z: features, as given by the model.
        """

        if self.mu is None:
            raise RequiresFittingException

        md_k = self._calc_gaussian_scores(z)
        md_0 = self._background_score(z)

        score = torch.min(md_k - md_0.view(-1, 1), dim=1).values
        return score

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor
        """
        if self.model is None:
            raise ModelNotSetException

        if self.eps > 0:
            x = self._odin_preprocess(x, x.device)

        features = self.model(x)
        return self.predict_features(features)
