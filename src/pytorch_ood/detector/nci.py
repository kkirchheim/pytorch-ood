"""

..  autoclass:: pytorch_ood.detector.NCI
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members:
"""

import logging

import torch
from torch import Tensor
from torch.nn import Linear, Module
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils import extract_features

log = logging.getLogger(__name__)


class NCI(FeaturesDetector):
    """
    Implements the Neural-Collapse Inspired OOD detector from the paper
    *Detecting Out-of-distribution through the Lens of Neural Collapse*.

    Computes a global mean :math:`\\mu_g` of all features from the fitting set to center representations during inference.
    Let :math:`h` be the representation of some input and :math:`z = h - \\mu_g` be the centered representation. The score is calculated as

    .. math::
        - \\frac{z \\cdot w_c}{\\lVert z \\rVert_2} - \\alpha \\lVert h \\rVert_1

    where :math:`w_c` is the weight vector for the class that the model predicted for the input, and :math:`\\alpha`
    is a hyperparameter that has to be tuned (see ``hyperparameter_space``).
    The score is the negated NCI, so that larger values indicate outliers.

    The first term will penalize inputs whose representation does not align with the class vectors,
    while the second term penalizes inputs whose representation resides close to the origin.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Detecting Out-of-Distribution Through the Lens of Neural Collapse",
            venue="CVPR",
            year=2025,
            url="https://arxiv.org/pdf/2311.01479",
            code="https://github.com/litianliu/NCI-OOD",
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    #: default search space for :class:`pytorch_ood.utils.GridSearch` (APS tuning of
    #: the feature-norm penalty weight against a held-out ID+OOD validation split)
    hyperparameter_space = {"alpha": [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]}

    def __init__(self, encoder: Module, head: Linear, alpha: float = 0.0) -> None:
        """
        :param encoder: model mapping inputs to features
        :param head: linear classification head of the model. A copy is stored; it is used to determine
            the predicted class and the weight vectors :math:`w_c`.
        :param alpha: weight for feature norm penalty. Will be ignored if :math:`\\leq 0`
        """
        import copy

        super(NCI, self).__init__()
        self.encoder = encoder
        self.head = copy.deepcopy(head)
        self.alpha = alpha
        self.global_mean = None

    def fit(self, data_loader) -> Self:
        """
        :param data_loader: data loader used to compute :math:`\\mu_g`. Labels are ignored, so it must
            only contain ID data.
        :return: self
        """
        if self.encoder is None:
            raise ModelNotSetException

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        z, y = extract_features(data_loader, self.encoder, device=device)

        return self.fit_features(z)

    def fit_features(self, z: torch.Tensor, *args, **kwargs) -> Self:
        """
        :param z: features :math:`h` of the fitting set, of shape :math:`N \\times D`, used to compute :math:`\\mu_g`.
            Additional arguments (e.g. labels) are ignored.
        :return: self
        """
        device = self.device or z.device
        z = z.detach().to(device).float()
        self.global_mean = z.mean(dim=0)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate outlier score for inputs, which will be passed through the encoder.

        :param x: input tensor, will be passed through model

        :return: outlier scores of shape :math:`B`
        """
        if self.encoder is None:
            raise ModelNotSetException

        return self.predict_features(self.encoder(x))

    def _cos(self, centered_features: Tensor, class_weight_vectors: Tensor) -> Tensor:
        # dot product between class vectors and centered features
        nom = (centered_features * class_weight_vectors).sum(dim=1)

        # l2 norm of feature vectors, the class_weight_vector norm term gets canceled
        denom = centered_features.pow(2).sum(dim=1).sqrt()
        return nom / denom

    @torch.no_grad()
    def predict_features(self, features: Tensor) -> Tensor:
        """
        Compute outlier scores based on features (without passing through encoder).

        :param features: features :math:`h` of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        """

        if self.global_mean is None:
            raise RequiresFittingException()

        device = self.device or features.device
        features = features.detach().to(device).float()
        self.head = self.head.to(device)
        self.global_mean = self.global_mean.to(device)

        centered_features = features - self.global_mean
        predicted_class = self.head(features).argmax(dim=1)
        class_weight_vectors = self.head.weight.data[predicted_class]

        p_score = self._cos(centered_features, class_weight_vectors)

        if self.alpha <= 0:
            return -p_score
        else:
            # TODO: add different options for p-norm, here we use l1
            feature_norm = features.abs().sum(dim=1)
            return -p_score - self.alpha * feature_norm
