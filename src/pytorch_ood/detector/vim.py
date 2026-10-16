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

from ..api import DetectorInfo, Paper, RequiresFittingException, Task
from .residual import Residual

log = logging.getLogger(__name__)


class ViM(Residual):
    """
    Implements Virtual Logit Matching (ViM) from the paper *ViM: Out-Of-Distribution with Virtual-logit Matching*.

    ViM decomposes the (centered) features into a principal subspace of dimension :math:`d` and
    its orthogonal residual. The residual norm, scaled by a factor :math:`\\alpha` estimated in ``fit``, is
    the *virtual logit*. The outlier score is the virtual logit minus the energy of the real logits:

    .. math::
        \\alpha \\lVert x^{P\\perp} \\rVert - \\log \\sum_i e^{l_i(x)}

    where :math:`x^{P\\perp}` is the residual of the features :math:`x` after centering at the origin
    :math:`u = -W^{+} b` and projecting out the principal subspace, :math:`W` and :math:`b` are the weights and
    biases of the last layer and :math:`l_i(x)` are the logits.

    ViM extends :class:`Residual`, which uses the unscaled residual norm as outlier score.

    .. rubric:: Examples

    .. code-block:: python

        model = WideResNet()
        detector = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
        detector.fit(train_loader)
        scores = detector(images)
    """

    info = DetectorInfo(
        paper=Paper(
            title="ViM: Out-Of-Distribution with Virtual-logit Matching",
            venue="CVPR",
            year=2022,
            url="https://arxiv.org/abs/2203.10807",
            code="https://github.com/haoqiwang/vim/",
        ),
        tasks={Task.CLASSIFICATION},
    )

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
        :param w: weights :math:`W` of the last layer of the network, of shape :math:`C \\times D`
        :param b: biases :math:`b` of the last layer of the network, of shape :math:`C`
        """
        super(ViM, self).__init__(encoder=encoder, d=d, w=w, b=b)
        self.alpha: Optional[float] = None  #: the computed :math:`\\alpha` value

    @torch.no_grad()
    def predict_features(self, x: Tensor) -> Tensor:
        """
        :param x: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        """
        if self.principal_subspace is None or self.alpha is None:
            raise RequiresFittingException()

        x = x.detach().to(self.w.device).float()
        logits = self._get_logits(x)  # (N, C)
        vlogit = self._residual_norm(x) * self.alpha  # (N,)

        # torch.logsumexp subtracts the max internally, so it is already numerically
        # stable without clamping -- clamping would distort the result for models
        # whose raw logits exceed +-100.
        energy = torch.logsumexp(logits, dim=-1)  # (N,)

        score = -vlogit + energy
        return -score

    def fit_features(self, features: Tensor, labels: Tensor) -> Self:
        """
        Computes the principal subspace and alpha from pre-extracted features. Ignores OOD samples.

        :param features: features of shape :math:`N \\times D`
        :param labels: class labels of shape :math:`N`
        :return: self
        """
        super(ViM, self).fit_features(features, labels)

        features = features.detach().to(self.w.device).float()
        logits = self._get_logits(features)  # (N, C)

        log.info("Computing alpha ...")
        vlogits = self._residual_norm(features)  # (N,)
        self.alpha = (logits.max(dim=-1).values.mean() / vlogits.mean()).item()
        log.info(f"{self.alpha=:.4f}")
        return self
