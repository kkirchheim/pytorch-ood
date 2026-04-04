"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.fDBD
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import TypeVar

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.utils.data import DataLoader

from ..api import FeaturesDetector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class fDBD(FeaturesDetector):
    """
    Implements the Fast Decision Boundary Distance detector from the paper
    *Fast Decision Boundary based Out-of-Distribution Detector*.

    Computes the closed-form distance from each sample's penultimate-layer features
    to the decision boundaries of the linear classification head, averaged over all
    non-predicted classes and normalized by the distance to the training feature mean.

    The score for a sample with features :math:`z` and predicted class :math:`\\hat{y}` is:

    .. math::
        - \\frac{1}{|C|-1} \\sum_{c \\neq \\hat{y}}
        \\frac{| \\text{logit}_{\\hat{y}} - \\text{logit}_c |}
        {\\lVert w_{\\hat{y}} - w_c \\rVert_2 \\cdot \\lVert z - \\mu \\rVert_2}

    where :math:`w_k` are the weight vectors of the classification head and :math:`\\mu`
    is the mean of training features. This method is hyperparameter-free.

    :see Paper: `ArXiv <https://arxiv.org/abs/2312.11536>`__
    :see Implementation: `GitHub <https://github.com/litianliu/fDBD-OOD>`__
    """

    requires_fit = True

    def __init__(self, encoder: Module, head: Linear) -> None:
        """
        :param encoder: model mapping inputs to penultimate-layer features
        :param head: the linear classification head of the model
        """
        super(fDBD, self).__init__()
        self.encoder = encoder
        self.head = head
        self.train_mean: Tensor = None
        self._denom_matrix: Tensor = None

    def _precompute_denom_matrix(self) -> None:
        """
        Precompute the pairwise weight-difference norms matrix.
        ``denom_matrix[p, c] = ||w_p - w_c||_2``, with diagonal set to 1 to avoid division by zero.
        """
        w = self.head.weight.data  # (num_classes, feature_dim)
        n_classes = w.shape[0]
        # w[i] - w[j] for all pairs via broadcasting
        diff = w.unsqueeze(1) - w.unsqueeze(0)  # (C, C, D)
        denom = diff.norm(dim=2)  # (C, C)
        # avoid division by zero on diagonal
        denom[torch.arange(n_classes), torch.arange(n_classes)] = 1.0
        self._denom_matrix = denom

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Compute the training feature mean :math:`\\mu`.

        :param data_loader: data loader with training data
        """
        if self.encoder is None:
            raise ModelNotSetException()

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device)
        return self.fit_features(z)

    def fit_features(self: Self, z: Tensor, *args, **kwargs) -> Self:
        """
        Compute the training feature mean directly from features.

        :param z: training features
        """
        self.train_mean = z.mean(dim=0)
        self._precompute_denom_matrix()
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through the encoder
        """
        if self.encoder is None:
            raise ModelNotSetException()

        z = self.encoder(x)
        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute outlier scores from features.

        :param z: penultimate-layer features
        :return: outlier scores (higher = more OOD)
        """
        if self.train_mean is None:
            raise RequiresFittingException()

        device = z.device
        z = z.float()
        head = self.head.to(device)
        train_mean = self.train_mean.to(device)
        denom_matrix = self._denom_matrix.to(device)

        # compute logits
        logits = head(z)  # (N, C)

        # predicted class and its logit value
        max_logits, pred_classes = logits.max(dim=1)  # (N,)

        # |logit_pred - logit_c| for all classes c
        logit_diff = torch.abs(logits - max_logits.unsqueeze(1))  # (N, C)

        # weight-difference norms for each sample's predicted class
        weight_norms = denom_matrix[pred_classes]  # (N, C)

        # per-boundary distances, summed over all classes (diagonal contributes 0 since logit_diff is 0 there)
        boundary_sum = (logit_diff / weight_norms).sum(dim=1)  # (N,)

        # normalize by distance to training mean
        feat_dist = (z - train_mean).norm(dim=1)  # (N,)
        # guard against zero-norm features
        feat_dist = feat_dist.clamp(min=1e-8)

        n_classes = logits.shape[1]
        score = boundary_sum / ((n_classes - 1) * feat_dist)

        # negate: higher original score = more ID, convention is higher = more OOD
        return -score
