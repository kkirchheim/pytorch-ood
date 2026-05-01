"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.NNGuide
    :members:

"""

import logging
from typing import Callable, Optional, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_ood.api import FeaturesDetector, ModelNotSetException, RequiresFittingException
from pytorch_ood.utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class NNGuide(FeaturesDetector):
    """
    Implements *Nearest Neighbor Guidance for Out-of-Distribution Detection*.

    Guides classifier-based scores using k-NN similarity to an energy-weighted
    feature bank. The feature bank is constructed by scaling in-distribution training
    features with their corresponding energy scores. At inference, the outlier score
    is the negated product of the k-NN guidance (mean inner product with the
    energy-scaled feature bank) and the sample's own energy:

    .. math::
        s(x) = - \\underbrace{\\frac{1}{k} \\sum_{z \\in \\mathcal{N}_k(x)}
        \\langle f(x),\\, E(z) \\cdot f(z) \\rangle}_{\\text{guidance}} \\cdot E(x)

    where :math:`E(x) = \\log \\sum_i \\exp(l_i(x))` is the energy score,
    :math:`f(x)` are the penultimate-layer features, and :math:`\\mathcal{N}_k(x)` are the
    :math:`k` nearest neighbors in the energy-scaled feature bank measured by inner product.

    The encoder extracts penultimate-layer features. The head computes logits from features
    and is used internally to compute energy scores, similar to :class:`ViM`.

    Example Code:

    .. code :: python

        model = WideResNet()
        detector = NNGuide(
            encoder=model.features,
            head=model.fc,
            k=10
        )
        detector.fit(train_loader)
        scores = detector(images)

    :see Paper: `arXiv <https://arxiv.org/abs/2309.14888>`__

    """

    requires_fit = True

    def __init__(
        self,
        encoder: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        k: int = 10,
    ):
        """
        :param encoder: neural network that extracts penultimate-layer features
        :param head: callable that maps features to logits (e.g., a linear layer)
        :param k: number of nearest neighbors for guidance (default: 10)
        """
        super(NNGuide, self).__init__()
        self.encoder = encoder
        self.head = head
        self.k = k
        self._scaled_features: Optional[Tensor] = None

        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError("You have to install scikit-learn to use this detector")

        self._nbrs = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)

    def fit(self: Self, data_loader: DataLoader, device=None) -> Self:
        """
        Extract features from the data loader and build the energy-scaled feature bank.

        :param data_loader: data loader with ID training data
        :param device: device for feature extraction. If ``None``, inferred from encoder.
        """
        if device is None:
            device = self.device
            if device is None:
                if isinstance(self.encoder, torch.nn.Module):
                    device = next(self.encoder.parameters()).device
                else:
                    device = "cpu"
                log.warning(f"No device given. Will use '{device}'.")

        if isinstance(self.encoder, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            self.encoder.to(device)

        z, y = extract_features(model=self.encoder, data_loader=data_loader, device=device)
        return self.fit_features(z, y)

    def fit_features(self: Self, z: Tensor, labels: Tensor) -> Self:
        """
        Build the energy-scaled feature bank from pre-extracted features.

        :param z: features, shape ``(n, feature_dim)``
        :param labels: corresponding labels
        """
        known = is_known(labels)

        if not known.any():
            raise ValueError("No ID samples")

        device = self.device or z.device
        z = z[known].detach().to(device).float()

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        with torch.no_grad():
            logits = self.head(z)
        energy = torch.logsumexp(logits, dim=1)

        # Normalize energy to avoid numerical issues from large scaling factors
        # Use z/||z|| to normalize features, then scale by normalized energy
        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        energy_norm = energy / (energy.mean() + 1e-8)
        self._scaled_features = z_norm * energy_norm[:, None]

        # Fit k-NN model on normalized features for efficient neighbor search
        self._nbrs.fit(self._scaled_features.detach().cpu().numpy())

        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model inputs
        """
        if self.encoder is None:
            raise ModelNotSetException()
        if self._scaled_features is None:
            raise RequiresFittingException()

        with torch.no_grad():
            z = self.encoder(x)
        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute the NNGuide outlier score from pre-extracted features.

        :param z: features, shape ``(batch, feature_dim)``
        """
        if self._scaled_features is None:
            raise RequiresFittingException()

        device = self.device or z.device

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        logits = self.head(z)
        energy = torch.logsumexp(logits, dim=1)

        # Normalize features for numerical stability
        z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-8)

        # Use efficient k-NN search with cosine distance to find k nearest neighbors
        distances, _ = self._nbrs.kneighbors(z_norm.detach().cpu().numpy(), n_neighbors=self.k)

        # Convert cosine distances to similarities: sim = 1 - distance
        similarities = 1.0 - distances

        # Mean similarity over k nearest neighbors
        guidance = torch.from_numpy(similarities.mean(axis=1)).to(device).float()

        # higher guidance * energy = more in-distribution, so negate for outlier score
        return -(guidance * energy)
