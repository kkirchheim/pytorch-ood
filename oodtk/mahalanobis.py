"""

..  autoclass:: oodtk.Mahalanobis
    :members:



"""
import logging

import torch

from oodtk.utils import TensorBuffer, contains_unknown, is_known, is_unknown

from .api import Method

log = logging.getLogger(__name__)


class Mahalanobis(torch.nn.Module, Method):
    """
    Implements the Mahalanobis Method.

    This method calculates a class center :math:`\\mu_y` for each class,
    and a shared covariance matrix :math:`\\Sigma` from the data.

    :see Implementation: https://github.com/pokaxpoka/deep_Mahalanobis_detector
    """

    def __init__(self, model: torch.nn.Module):
        """

        :param model: the Neural Network
        """
        super(Mahalanobis, self).__init__()
        self.model = model
        self.mu = None
        self.cov = None
        self.precision = None

    def fit(self, data_loader):
        """
        Fit parameters of the multi variate gaussian

        :param data_loader: dataset to fit on.
        :return:
        """
        self.model.eval()

        # TODO: quickfix
        dev = list(self.model.parameters())[0].device

        # TODO: add option to buffer to GPU
        buffer = TensorBuffer()
        log.debug("Extracting features")
        for batch in data_loader:
            x, y = batch
            x = x.to(dev)
            y = y.to(dev)
            known = is_known(y)
            z = self.model(x[known])
            # flatten
            x = z.view(x.shape[0], -1)
            buffer.append("embedding", z[known])
            buffer.append("label", y[known])

        z = buffer.get("embedding")
        y = buffer.get("label")
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
            zs = z[idxs]
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).T.mm(zs - self.mu[clazz])

        self.precision = torch.linalg.inv(self.cov)
        buffer.clear()

    def predict(self, x) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        z = self.model(x)

        # TODO: original uses mean over feature maps
        # here, we just flatten
        z = self.model(x)
        z = z.view(z.shape[0], -1)

        score = 0

        for clazz in range(self.n_classes):
            zero_f = z.data - self.mu[clazz]
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()

            if clazz == 0:
                score = term_gau.view(-1, 1)
            else:
                score = torch.cat((score, term_gau.view(-1, 1)), dim=1)

        return score.max(dim=1).values

    @property
    def n_classes(self):
        return self.mu.shape[0]
