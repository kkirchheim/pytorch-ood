"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.PNML
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..api import FeaturesDetector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features, is_known

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class PNML(FeaturesDetector):
    """
    Implements the pNML regret from
    *Single Layer Predictive Normalized Maximum Likelihood for Out-of-Distribution Detection*.

    Uses normalized penultimate-layer features together with the classifier probabilities
    to compute the pNML regret. Higher regret indicates a more likely OOD sample.

    For normalized training features :math:`X`, their Moore-Penrose pseudoinverse
    :math:`X^+`, normalized test feature :math:`z`, classifier probabilities
    :math:`p_i(z)`, and
    :math:`\\kappa(z) = \\frac{z^\\top X^+ X^{+\\top} z}{1 + z^\\top X^+ X^{+\\top} z}`,
    the detector scores a sample by

    .. math::
        R(z) = \\frac{1}{\\log C}
        \\log \\sum_{i=1}^{C}
        \\frac{p_i(z)}{p_i(z) + (1 - p_i(z)) p_i(z)^{\\kappa(z)}}.

    Intuitively, the score is low when a sample is well supported by the training feature geometry
    and the classifier is confident, and high when the sample falls in weakly supported directions.

    :see Paper:
        `ArXiv <https://arxiv.org/abs/2110.09246>`__
    :see Implementation:
        `GitHub <https://github.com/kobybibas/pnml_ood_detection>`__
    """

    requires_fit = True

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        head: Optional[Callable[[Tensor], Tensor]],
        eps: float = 1e-12,
    ):
        """
        :param encoder: feature encoder mapping inputs to penultimate-layer features
        :param head: classification head mapping normalized features to logits
        :param eps: numerical stability constant for probability clamping
        """
        self.encoder = encoder
        self.head = head
        self.eps = eps
        self._feature_projector = None
        self._log_num_classes = None

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extract features and fit the pNML detector.

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
        return self.fit_features(z, y)

    def fit_features(self: Self, z: Tensor, labels: Tensor) -> Self:
        """
        Fit pNML directly on penultimate-layer features.

        Labels are only used to filter out OOD-marked samples when present.

        :param z: training features
        :param labels: class labels
        """
        known = is_known(labels)
        if not known.any():
            raise ValueError("No ID samples found.")

        target_device = self.device or z.device

        z = z[known].detach().to(target_device).float()
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        # X^+ (X^+)^T == pinv(X^T X) for any X, and equals (X^T X)^{-1} exactly whenever
        # X^T X is invertible (the practically-always-true case for real feature data).
        # Computing it this way -- pinv on the small (D, D) Gram matrix -- instead of
        # pinv(X) directly avoids an SVD over the full (N, D) training-feature matrix,
        # which is intractable at full-dataset scale (e.g. N=1.28M for ImageNet-1K)
        # even though D is typically just a few thousand.
        self._feature_projector = torch.linalg.pinv(z.T @ z)
        self._log_num_classes = None
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor, will be passed through the backbone
        """
        if self.encoder is None:
            raise ModelNotSetException()

        z = self.encoder(x)
        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculate outlier scores using the normalized pNML regret.

        :param z: penultimate-layer features
        :return: outlier scores (higher = more OOD)
        """
        if self.head is None:
            raise ModelNotSetException(msg="When using predict_features(), head must not be None")

        if self._feature_projector is None:
            raise RequiresFittingException()

        device = self._feature_projector.device
        z = z.detach().to(device).float()
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        probs = torch.softmax(self.head(z), dim=1)
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        x_proj = (z @ self._feature_projector * z).sum(dim=1)
        x_t_g = x_proj / (1.0 + x_proj)

        regret_terms = probs / (probs + (1.0 - probs) * probs.pow(x_t_g.unsqueeze(1)))
        regret = torch.log(regret_terms.sum(dim=1))

        if self._log_num_classes is None:
            self._log_num_classes = torch.log(torch.tensor(float(probs.shape[1]), device=device))

        return regret / self._log_num_classes
