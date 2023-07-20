"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.KNN
    :members:

"""
import logging
from typing import TypeVar, Callable

from torch import Tensor, tensor
from torch.utils.data import DataLoader

from pytorch_ood.api import RequiresFittingException, Detector, ModelNotSetException
from pytorch_ood.utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class KNN(Detector):
    """
    Implements the detector from the paper
    *Out-of-Distribution Detection with Deep Nearest Neighbors*.

    Fits a nearest neighbor model to the IN samples an uses the distance
    from the nearest neighbor as outlier score:

    .. math:: \\min_{z \\in \\mathcal{D}} \\lVert f(x) - f(z) \\rVert_2

    where :math:`\\mathcal{D}` is the dataset used to train the nearest neighbor model.

    The original paper found that using contrastive pre-training could increase the performance.

    :see PMLR: `arXiv <https://proceedings.mlr.press/v162/sun22d.html>`__
    """
    def __init__(self, model: Callable[[Tensor], Tensor], **knn_kwargs):
        """
        :param model: neural network to use
        :param knn_kwargs: dict with keyword arguments that will be passed to the scikit learns k-NN
        """
        self.model = model
        self._is_fitted = False

        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise Exception(f"You have to install scikit-learn to use this detector")

        self.knn: NearestNeighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1, **knn_kwargs)

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: inputs, will be passed through model
        """
        if not self.model:
            raise ModelNotSetException()

        z = self.model(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        :param z: features
        :param k: number of neighbors
        """

        if not self._is_fitted:
            raise RequiresFittingException()

        dist, idx = self.knn.kneighbors(z.detach().numpy(), n_neighbors=1, return_distance=True)

        return tensor(dist)

    def fit_features(self: Self, z: Tensor, labels: Tensor) -> Self:
        """
        Fits nearest neighbor model. Ignores OOD inputs.

        :param z: features
        :param labels: labels for features
        """
        known = is_known(labels)

        if not known.any():
            raise ValueError(f"No IN samples")

        self.knn.fit(z[known].numpy())

        self._is_fitted = True

        return self

    def fit(self: Self, loader: DataLoader, device: str = "cpu") -> Self:
        """
        Extracts features and fits the kNN-Model

        :param loader: data loader
        :param device: device used for extracting logits
        """
        z, y = extract_features(model=self.model, data_loader=loader, device=device)
        return self.fit_features(z, y)
