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

from pytorch_ood.api import Detector, ModelNotSetException, RequiresFittingException
from pytorch_ood.utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class NNGuide(Detector):
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

    The model passed to the constructor should extract penultimate-layer features.
    The classification head weights ``w`` and biases ``b`` are used internally to
    compute logits from features, similar to :class:`ViM`.

    :see Paper: `arXiv <https://arxiv.org/abs/2309.14888>`__

    """

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        w: Tensor,
        b: Tensor,
        k: int = 10,
    ):
        """
        :param model: neural network that extracts penultimate-layer features
        :param w: weight matrix of the classification head, shape ``(num_classes, feature_dim)``
        :param b: bias vector of the classification head, shape ``(num_classes,)``
        :param k: number of nearest neighbors for guidance (default: 10)
        """
        super(NNGuide, self).__init__()
        self.model = model
        self.w = w.detach().cpu().float()
        self.b = b.detach().cpu().float()
        self.k = k
        self._scaled_features: Optional[Tensor] = None

    def _logits(self, features: Tensor) -> Tensor:
        """Compute logits from features using the stored classification head."""
        return features @ self.w.T + self.b

    def fit(self: Self, data_loader: DataLoader, device=None) -> Self:
        """
        Extract features from the data loader and build the energy-scaled feature bank.

        :param data_loader: data loader with ID training data
        :param device: device for feature extraction. If ``None``, inferred from model.
        """
        if device is None:
            if isinstance(self.model, torch.nn.Module):
                device = next(self.model.parameters()).device
            else:
                device = "cpu"
            log.warning(f"No device given. Will use '{device}'.")

        if isinstance(self.model, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            self.model.to(device)

        z, y = extract_features(model=self.model, data_loader=data_loader, device=device)
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

        z = z[known].cpu().float()
        logits = self._logits(z)
        energy = torch.logsumexp(logits, dim=1)
        self._scaled_features = z * energy[:, None]

        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model inputs
        """
        if self.model is None:
            raise ModelNotSetException()
        if self._scaled_features is None:
            raise RequiresFittingException()

        with torch.no_grad():
            z = self.model(x)
        return self.predict_features(z)

    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute the NNGuide outlier score from pre-extracted features.

        :param z: features, shape ``(batch, feature_dim)``
        """
        if self._scaled_features is None:
            raise RequiresFittingException()

        z = z.detach().cpu().float()
        logits = self._logits(z)
        energy = torch.logsumexp(logits, dim=1)

        # inner product between test features and energy-scaled training features
        # shape: (batch, n_train)
        sim = z @ self._scaled_features.T

        # mean inner product over k nearest neighbors (largest inner products)
        topk_sim, _ = sim.topk(self.k, dim=1)
        guidance = topk_sim.mean(dim=1)

        # higher guidance * energy = more in-distribution, so negate for outlier score
        return -(guidance * energy)
