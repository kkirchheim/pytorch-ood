"""
..  autoclass:: pytorch_ood.detector.Mahalanobis
    :members:
"""
import logging
import warnings
from typing import Callable, List, Optional, TypeVar

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..api import Detector, RequiresFittingException
from ..utils import TensorBuffer, contains_unknown, is_known, is_unknown

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class Mahalanobis(Detector):
    """
    Implements the Mahalanobis Method from the paper *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks*.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.

    Also uses ODIN preprocessing.

    :see Implementation: `GitHub <https://github.com/pokaxpoka/deep_Mahalanobis_detector>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1807.03888>`__
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
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
        self.mu: torch.Tensor = None  #: Centers
        self.cov: torch.Tensor = None  #: Covariance Matrix
        self.precision: torch.Tensor = None  #: Precision Matrix
        self.eps: float = eps  #: epsilon
        self.norm_std = norm_std

    @staticmethod
    def _extract(data_loader, model, device):
        """
        Extract embeddings from model

        .. note :: this should be moved into a dedicated function

        :param data_loader:
        :param model:
        :param device:
        :return:
        """
        # TODO: add option to buffer to GPU
        buffer = TensorBuffer()

        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            known = is_known(y)
            if known.any():
                z = model(x[known])
                # flatten
                z = z.view(known.sum(), -1)
                buffer.append("embedding", z)
                buffer.append("label", y[known])

        if buffer.is_empty():
            raise ValueError("No IN instances in loader")

        z = buffer.get("embedding")
        y = buffer.get("label")

        buffer.clear()
        return z, y

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

        z, y = Mahalanobis._extract(data_loader, self.model, device)

        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        n_classes = len(classes)
        self.mu = torch.zeros(size=(n_classes, z.shape[-1])).to(device)
        self.cov = torch.zeros(size=(z.shape[-1], z.shape[-1])).to(device)

        for clazz in range(n_classes):
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs].to(device)
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).T.mm(zs - self.mu[clazz])

        self.precision = torch.linalg.inv(self.cov)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor
        """
        if self.mu is None:
            raise RequiresFittingException

        dev = x.device

        if self.eps > 0:
            x = self._odin_preprocess(x, dev)

        features = self.model(x)
        features = features.view(features.size(0), features.size(1), -1)
        features = torch.mean(features, 2)
        noise_gaussian_score = 0

        for clazz in range(self.n_classes):
            # batch_sample_mean = sample_mean[layer_index][i]
            centered_features = features.data - self.mu[clazz]
            term_gau = (
                -0.5
                * torch.mm(
                    torch.mm(centered_features, self.precision), centered_features.t()
                ).diag()
            )
            if clazz == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score = torch.max(noise_gaussian_score, dim=1).values
        return -noise_gaussian_score

    def _odin_preprocess(self, x, dev):
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
                            torch.mm(centered_features, self.precision), centered_features.t()
                        ).diag()
                    )

                    if clazz == 0:
                        score = term_gau.view(-1, 1)
                    else:
                        score = torch.cat((score, term_gau.view(-1, 1)), dim=1)

                # Input_processing
                # calculate gradient of inputs with respect to score of predicted class, according to mahalanobis distance
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
        return self.mu.shape[0]
