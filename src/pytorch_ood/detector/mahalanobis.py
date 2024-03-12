"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.Mahalanobis
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

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class Mahalanobis(Detector):
    """
    Implements the Mahalanobis Method from the paper *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks*.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.
    The outlier scores are then calculated as

    .. math :: - \\max_k \\lbrace (f(x) - \\mu_k)^{\\top} \\Sigma^{-1} (f(x) - \\mu_k) \\rbrace

    Also uses ODIN preprocessing.

    :see Implementation: `GitHub <https://github.com/pokaxpoka/deep_Mahalanobis_detector>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1807.03888>`__
    """

    def __init__(
            self,
            model: Callable[[Tensor], Tensor],
            eps: float = 0.002,
            norm_std: Optional[List] = None,
    ):
        """
        :param model: the Neural Network, should output features
        :param eps: magnitude for gradient based input preprocessing
        :param norm_std: Standard deviations for input normalization
        """
        super(Mahalanobis, self).__init__()
        self.model = model
        self.mu: Tensor = None  #: Centers
        self.cov: Tensor = None  #: Covariance Matrix
        self.precision: Tensor = None  #: Precision Matrix
        self.eps: float = eps  #: epsilon
        self.norm_std = norm_std

    def fit(self: Self, data_loader: DataLoader, device: str = None) -> Self:
        """
        Fit parameters of the multi variate gaussian.

        :param data_loader: dataset to fit on.
        :param device: device to use
        :return:
        """
        if device is None:
            device = list(self.model.parameters())[0].device
            log.warning(f"No device given. Will use '{device}'.")

        z, y = extract_features(data_loader, self.model, device)
        return self.fit_features(z, y, device)

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

        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        n_classes = len(classes)
        self.mu = torch.zeros(size=(n_classes, z.shape[-1]), device=device)
        self.cov = torch.zeros(size=(z.shape[-1], z.shape[-1]), device=device)

        for clazz in range(n_classes):
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs]
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).T.mm(zs - self.mu[clazz])

        self.cov += torch.eye(self.cov.shape[0], device=self.cov.device) * 1e-6
        self.precision = torch.linalg.inv(self.cov)
        return self

    def _calc_gaussian_scores(self, z: Tensor) -> Tensor:
        """

        """
        features = z.view(z.size(0), z.size(1), -1)
        features = torch.mean(features, 2)
        md_k = []

        # calculate per class scores
        for clazz in range(self.n_classes):
            centered_z = features.data - self.mu[clazz]
            term_gau = -0.5 * torch.mm(torch.mm(centered_z, self.precision), centered_z.t()).diag()
            md_k.append(term_gau.view(-1, 1))

        return torch.cat(md_k, 1)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculates mahalanobis distance directly on features.
        ODIN preprocessing will not be applied.

        :param z: features, as given by the model.
        """
        if self.mu is None:
            raise RequiresFittingException

        md_k = self._calc_gaussian_scores(z)
        score = - torch.max(md_k, dim=1).values
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

    def _odin_preprocess(self, x: Tensor, dev: str):
        """
        NOTE: the original implementation uses mean over feature maps. here, we just flatten
        """
        # does not work in inference mode, this sometimes collides with pytorch-lightning
        if torch.is_inference_mode_enabled():
            warnings.warn("ODIN not compatible with inference mode. Will be deactivated.")

        with torch.inference_mode(False):
            if torch.is_inference(x):
                x = x.clone()

            with torch.enable_grad():
                x = Variable(x, requires_grad=True)
                features = self.model(x)
                features = features.view(features.shape[0], -1)  # flatten
                score = None

                for clazz in range(self.n_classes):
                    centered_features = features.data - self.mu[clazz]
                    term_gau = (
                            -0.5
                            * torch.mm(
                        torch.mm(centered_features, self.precision),
                        centered_features.t(),
                    ).diag()
                    )

                    if clazz == 0:
                        score = term_gau.view(-1, 1)
                    else:
                        score = torch.cat((score, term_gau.view(-1, 1)), dim=1)

                # calculate gradient of inputs with respect to score of predicted class,
                # according to mahalanobis distance
                sample_pred = score.max(dim=1).indices
                batch_sample_mean = self.mu.index_select(0, sample_pred)
                centered_features = features - Variable(batch_sample_mean)
                pure_gau = (
                        -0.5
                        * torch.mm(
                    torch.mm(centered_features, Variable(self.precision)),
                    centered_features.t(),
                ).diag()
                )
                loss = torch.mean(-pure_gau)
                loss.backward()

                gradient = torch.sign(x.grad.data)

        if self.norm_std:
            for i, std in enumerate(self.norm_std):
                gradient.index_copy_(
                    1,
                    torch.LongTensor([i]).to(dev),
                    gradient.index_select(1, torch.LongTensor([i]).to(dev)) / std,
                )
        perturbed_x = x.data - self.eps * gradient

        return perturbed_x

    @property
    def n_classes(self):
        """
        Number of classes the model is fitted for
        """
        if self.mu is None:
            raise RequiresFittingException

        return self.mu.shape[0]
