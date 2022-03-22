"""

..  autoclass:: oodtk.Mahalanobis
    :members:

"""
import logging

import torch

from oodtk.utils import TensorBuffer

from .api import Method

log = logging.getLogger(__name__)


class Mahalanobis(torch.nn.Module, Method):
    """
    Implements the Mahalanobis Method.

    This method calculates a class center :math:`\\mu_y` for each class,
     and a shared covariance matrix :math:`\\Sigma`.
    from the data.

    """

    def __init__(self, model: torch.nn.Module):
        """

        :param model: the Neural Network
        """
        super(Mahalanobis, self).__init__()
        self.model = model

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
            z = self.model(x)
            buffer.append("embedding", z)
            buffer.append("label", y)
            break

        z = buffer.get("embedding")
        y = buffer.get("label")

        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()
        n_classes = classes.max()
        # TODO: move to device
        self.mu = torch.zeros(size=(n_classes,))
        self.cov = torch.zeros(size=(z.shape))

        for n, clazz in enumerate(classes):
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs]
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).dot((zs - self.mu[clazz]).T)

    def predict(self, data_loader) -> torch.Tensor:
        pass

    def test(self, data_loader) -> dict:
        pass
