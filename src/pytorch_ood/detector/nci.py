"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.NCI
    :members:
    :exclude-members:
"""

from typing import TypeVar

import torch
from torch import Tensor
from torch.nn import Module, Linear

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features


Self = TypeVar("Self")


class NCI(Detector):
    """
    Implements the Neural-Collapse Inspired OOD detector from the paper
    *Detecting Out-of-distribution through the Lens of Neural Collapse*.

    Computes a global mean :math:`\\mu_g` of all features from the fitting set to center representations during inference.
    Let :math:`h` be the representation of some input and :math:`z = h - \\mu_g` be the centered representation. The score is calculated as

    .. math::
        - \\frac{z \\cdot w_c}{\\lVert z \\rVert_2} - \\alpha \\lVert h \\rVert_1

    where :math:`w_c` is the weight vector for the class that the model predicted for the input, and :math:`\\alpha`
    is a hyper parameter that has to be determined manually.

    The first term will penalize inputs whose representation does not align with the class vectors,
    while the second term penalizes inputs whose representation resides close to the origin.

    :see Paper:
        `CVPR <https://arxiv.org/pdf/2311.01479>`__

    :see Implementation:
        `GitHub <https://github.com/litianliu/NCI-OOD>`__

    """

    def __init__(self, encoder: Module, head: Linear, alpha: float = 0.0) -> None:
        """
        :param encoder: model mapping inputs to features
        :param head: the classification head of the model
        :param alpha: weight for feature norm penalty. Will be ignored if :math:`\\leq 0`
        """
        import copy

        super(NCI, self).__init__()
        self.encoder = encoder
        self.head = copy.deepcopy(head)
        self.alpha = alpha
        self.global_mean = None

    def fit(self: Self, data_loader) -> Self:
        """
        :param data_loader: data loader used to compute :math:`\\mu_g`
        """
        # fit global mean of features
        device = next(iter(self.encoder.parameters())).device

        z, y = extract_features(data_loader, self.encoder, device=device)

        return self.fit_features(z)

    def fit_features(self: Self, z: torch.Tensor, *args, **kwargs) -> Self:
        """
        :param z: input features used to compute :math:`\\mu_g`
        """
        self.global_mean = z.mean(dim=0)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate outlier score for inputs, which will be passed through the encoder.

        :param x: input tensor, will be passed through model

        :return: outlier score
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

        :param features: features given by the model
        """

        if self.global_mean is None:
            raise RequiresFittingException()

        features = features.cpu().float()
        self.head = self.head.cpu()
        self.global_mean = self.global_mean.cpu()

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
