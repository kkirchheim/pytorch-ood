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

        # TODO: add option to buffer to GPU
        buffer = TensorBuffer()
        log.debug("Extracting features")
        for batch in data_loader:
            x, y = batch
            known = is_known(y)
            z = self.model(x[known])
            buffer.append("embedding", z[known])
            buffer.append("label", y[known])

        z = buffer.get("embedding")
        y = buffer.get("label")
        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max()
        assert not contains_unknown(classes)

        n_classes = classes.max()
        # TODO: move to device
        self.mu = torch.zeros(size=(n_classes,))
        self.cov = torch.zeros(size=z.shape)

        for clazz in range(classes):
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs]
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).dot((zs - self.mu[clazz]).T)

        self.precision = torch.linalg.inv(self.cov)
        buffer.clear()

    def predict(self, x) -> torch.Tensor:
        return self.forward(x)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        z = self.model.intermediate_forward(x)

        # mean over feature maps
        z = z.view(z.size(0), z.size(1), -1)
        z = torch.mean(z, 2)

        score = 0

        for clazz in range(self.n_classes):
            zero_f = z.data - self.mu[clazz]
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()

            if clazz == 0:
                score = term_gau.view(-1, 1)
            else:
                score = torch.cat((score, term_gau.view(-1, 1)), 1)

        return score

    @property
    def n_classes(self):
        return self.mu.shape[0]
