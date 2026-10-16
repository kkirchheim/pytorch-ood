"""

..  autoclass:: pytorch_ood.detector.ViM
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional

import torch
from torch import Tensor
from typing_extensions import Self

from ..api import (
    ModelNotSetException,
    RequiresFittingException,
)

from .residual import Residual

log = logging.getLogger(__name__)


class ViM(Residual):
    """
    Implements Virtual Logit Matching (ViM) from the paper *ViM: Out-Of-Distribution with Virtual-logit Matching*.

    ViM decomposes the (centered) features into a principal subspace of dimension :math:`d` and
    its orthogonal residual. The residual norm, scaled by a factor :math:`\alpha` estimated in ``fit``, is
    the *virtual logit*. The outlier score is the virtual logit minus the energy of the real logits:

    .. math::
        \alpha \lVert x^{P\perp} \rVert - \log \sum_i e^{l_i(x)}

    where :math:`x^{P\perp}` is the residual of the features :math:`x` after centering at the origin
    :math:`u = -W^{+} b` and projecting out the principal subspace, :math:`W` and :math:`b` are the weights and
    biases of the last layer and :math:`l_i(x)` are the logits.

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
        detector.fit(train_loader)
        scores = detector(images)
    """

    def __init__(
        self,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]],
        d: int,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        :param encoder: feature encoder. Can be
            ``None`` when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param d: dimensionality of the principal subspace
        :param w: weights :math:`W` of the last layer of the network, of shape :math:`C \times D`
        :param b: biases :math:`b` of the last layer of the network, of shape :math:`C`
        """
        super(ViM, self).__init__(encoder=encoder, d=d, w=w, b=b)
        self.alpha: Optional[float] = None  #: the computed :math:`\alpha` value

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model input, will be passed through neural network
        :return: outlier scores of shape :math:`B`
        """
        if self.encoder is None:
            raise ModelNotSetException

        if self.principal_subspace is None or self.alpha is None:
            raise RequiresFittingException()

        with torch.no_grad():
            features = self.encoder(x)

        return self.predict_features(features)

    @torch.no_grad()
    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features of shape :math:`B \times D`
        :return: outlier scores of shape :math:`B`
        """
        device = self.w.device
        x = x.detach().to(device).float()
        logits = self._get_logits(x)  # (N, C)
        
        vlogit = super().predict_features(x) * self.alpha

        energy = torch.logsumexp(logits, dim=-1)  # (N,)

        score = -vlogit + energy
        return -score

    def fit_features(self, features: Tensor, labels: Tensor) -> Self:
        """
        Computes the principal subspace and alpha from pre-extracted features. Ignores OOD samples.

        :param features: features of shape :math:`N \times D`
        :param labels: class labels of shape :math:`N`
        :return: self
        """
        super().fit_features(features, labels)

        device = self.device or self.w.device
        features = features.detach().to(device).float()
        logits = self._get_logits(features)  # (N, C)

        log.info("Computing alpha ...")
        vlogits = super().predict_features(features)
        self.alpha = (logits.max(dim=-1).values.mean() / vlogits.mean()).item()
        log.info(f"{self.alpha=:.4f}")
        return self
