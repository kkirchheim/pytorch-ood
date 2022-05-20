"""
..  autoclass:: pytorch_ood.detector.Mahalanobis
    :members:
"""
import logging

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..api import Detector, RequiresFitException
from ..utils import TensorBuffer, contains_unknown, is_known, is_unknown

log = logging.getLogger(__name__)


class Mahalanobis(torch.nn.Module, Detector):
    """
    Implements the Mahalanobis Method.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.

    :see Implementation: https://github.com/pokaxpoka/deep_Mahalanobis_detector
    """

    def __init__(self, model: torch.nn.Module, eps=0.002, norm_std=None):
        """
        :param model: the Neural Network
        :param eps: magnitude for gradient based input preprocessing
        :param norm_std: Standard deviations for input normalization
        """
        super(Mahalanobis, self).__init__()
        self.model = model
        self.mu = None
        self.cov = None
        self.precision = None
        self.eps = eps
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
        log.debug("Extracting features")
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            known = is_known(y)
            z = model(x[known])
            # flatten
            x = z.view(x.shape[0], -1)
            buffer.append("embedding", z[known])
            buffer.append("label", y[known])

        z = buffer.get("embedding")
        y = buffer.get("label")

        buffer.clear()
        return z, y

    def fit(self, data_loader):
        """
        Fit parameters of the multi variate gaussian

        :param data_loader: dataset to fit on.
        :return:
        """
        self.model.eval()

        if isinstance(data_loader, DataLoader):
            # TODO: quickfix
            dev = list(self.model.parameters())[0].device
            z, y = Mahalanobis._extract(data_loader, self.model, dev)
        else:
            # TODO: implement initialization with raw data
            raise ValueError()

        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        n_classes = len(classes)
        self.mu = torch.zeros(size=(n_classes, z.shape[-1])).to(dev)
        self.cov = torch.zeros(size=(z.shape[-1], z.shape[-1])).to(dev)

        for clazz in range(n_classes):
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs].to(dev)
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).T.mm(zs - self.mu[clazz])

        self.precision = torch.linalg.inv(self.cov)
        return self

    def predict(self, x) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        if self.mu is None:
            raise RequiresFitException

        # TODO: quickfix
        dev = list(self.model.parameters())[0].device

        with torch.enable_grad():
            # TODO: original uses mean over feature maps

            x = Variable(x, requires_grad=True)

            # here, we just flatten
            z = self.model(x)
            z = z.view(z.shape[0], -1)
            score = None

            for clazz in range(self.n_classes):
                centered_z = z.data - self.mu[clazz]
                term_gau = (
                    -0.5 * torch.mm(torch.mm(centered_z, self.precision), centered_z.t()).diag()
                )

                if clazz == 0:
                    score = term_gau.view(-1, 1)
                else:
                    score = torch.cat((score, term_gau.view(-1, 1)), dim=1)

            # Input_processing
            # calculate gradient of inputs with respect to score of predicted class, according to mahalanobis distance
            sample_pred = score.max(dim=1).indices
            batch_sample_mean = self.mu.index_select(0, sample_pred)
            centered_z = z - Variable(batch_sample_mean)
            pure_gau = (
                -0.5
                * torch.mm(torch.mm(centered_z, Variable(self.precision)), centered_z.t()).diag()
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

        tempInputs = x.data - self.eps * gradient

        with torch.no_grad():
            noise_z = self.model(tempInputs)

        noise_z = noise_z.view(noise_z.size(0), noise_z.size(1), -1)
        noise_z = torch.mean(noise_z, 2)
        noise_gaussian_score = 0

        for clazz in range(self.n_classes):
            # batch_sample_mean = sample_mean[layer_index][i]
            centered_z = noise_z.data - self.mu[clazz]
            term_gau = -0.5 * torch.mm(torch.mm(centered_z, self.precision), centered_z.t()).diag()
            if clazz == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score = torch.max(noise_gaussian_score, dim=1).values
        return -noise_gaussian_score

    @property
    def n_classes(self):
        return self.mu.shape[0]
