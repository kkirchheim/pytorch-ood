"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge
.. image:: https://img.shields.io/badge/AI_Coded-yes-blue?style=flat-square
   :alt: slop-badge

..  autoclass:: pytorch_ood.detector.CADRef
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional, TypeVar

import torch
from torch import Tensor
from torch.nn import Linear

from ..api import RequiresFittingException
from .caref import CARef
from .energy import EnergyBased
from .maxlogit import MaxLogit
from .softmax import MaxSoftmax

Self = TypeVar("Self")
log = logging.getLogger(__name__)


def _gen_score(logits: Tensor, gamma: float = 0.1, M: Optional[int] = None) -> Tensor:
    """
    Generalized entropy as used for error scaling by the reference implementation of
    CADRef, as an outlier score:

    .. math::
        -\\left( \\sum_{j=1}^{M} p_j^{\\gamma} (1 - p_j)^{\\gamma} \\right)^{-1}

    over the :math:`M` largest softmax probabilities. :class:`CADRef` negates it, which
    recovers the reference's divisor :math:`1/\\sum_j p_j^{\\gamma}(1-p_j)^{\\gamma}`.
    This differs from
    :meth:`pytorch_ood.detector.GEN.score`, which averages rather than sums, clamps the
    probabilities away from 0 and 1, does not truncate by default, and is the entropy
    itself rather than its reciprocal. All of those are irrelevant for a ranked score but
    not for a divisor -- the clamp alone shifts it by several percent for confidently
    classified inputs -- so :meth:`~pytorch_ood.detector.GEN.score` must not be passed to :class:`CADRef`.

    Table 1 of the paper defines GEN as :math:`-\\sum_j p_j^{\\gamma}(1-p_j)^{\\gamma}`,
    which used literally as :math:`\\mathcal{S}_{logit}` in Equation (10) would be a
    negative divisor; the reference's reciprocal is a variant of it, not Table 1.

    :param logits: logits of input, shape ``(N, C)``
    :param gamma: power-transform exponent :math:`\\gamma`
    :param M: number of largest softmax probabilities to sum over. ``None`` (default) uses
        the reference's :math:`\\max(10, C/10)`, which gives 10 for CIFAR and 100 for
        ImageNet-1k. Note this differs from
        :meth:`~pytorch_ood.detector.GEN.score`, where ``None`` means all.
    """
    if M is None:
        M = max(10, logits.shape[1] // 10)

    p = logits.softmax(dim=1).sort(dim=1, descending=True).values[:, :M]
    entropy = (p.pow(gamma) * (1 - p).pow(gamma)).sum(dim=1)
    # the clamp only fires for saturated softmax outputs, where the reference divides by zero
    return -1.0 / entropy.clamp(min=CARef._NORM_EPS)


class CADRef(CARef):
    """
    Implements CADRef from the paper
    *CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature
    Leveraging*.

    CADRef refines :class:`CARef` by splitting the relative feature
    :math:`f(x) - \\mu^{\\hat{y}}` into the part that increases the maximum logit and the part
    that decreases it, and rescaling the two parts differently.

    Let :math:`\\hat{y}` be the predicted class, :math:`\\mu^{\\hat{y}}` the average training
    feature of that class (see :class:`CARef`), and :math:`w^{\\hat{y}}` the corresponding row
    of the weight matrix of the classification head. A feature dimension increases the maximum
    logit relative to the class mean exactly if the signs of :math:`w^{\\hat{y}}_i` and
    :math:`f(x)_i - \\mu^{\\hat{y}}_i` agree, which gives the two relative errors

    .. math::
        \\mathcal{E}_p(x) = \\frac{1}{\\lVert f(x) \\rVert_1}
        \\sum_i \\max\\left( \\operatorname{sign}(w^{\\hat{y}}_i)
        \\left( f(x)_i - \\mu^{\\hat{y}}_i \\right), 0 \\right)

    .. math::
        \\mathcal{E}_n(x) = \\frac{1}{\\lVert f(x) \\rVert_1}
        \\sum_i \\max\\left( -\\operatorname{sign}(w^{\\hat{y}}_i)
        \\left( f(x)_i - \\mu^{\\hat{y}}_i \\right), 0 \\right)

    which decompose the CARef error exactly, :math:`\\mathcal{E}_p + \\mathcal{E}_n = s_{CARef}`,
    up to dimensions with zero weight. The positive error separates ID from OOD poorly for
    confidently classified inputs, so it is divided by a per-sample logit-based confidence
    :math:`\\mathcal{S}(x)`, while the negative error is divided by the average confidence
    :math:`\\bar{\\mathcal{S}}` of the in-distribution training data, estimated during fitting:

    .. math::
        s(x) = \\frac{\\mathcal{E}_p(x)}{\\mathcal{S}(x)}
        + \\frac{\\mathcal{E}_n(x)}{\\bar{\\mathcal{S}}}

    Equations (7) and (8) of the paper split on
    :math:`\\operatorname{sign}(w^{max}_i \\cdot f(x)_i)`, i.e. over the raw feature, and write
    the numerator as the :math:`l_1` norm of a *sum* over the index set. This implementation
    follows the official code, which splits over the *relative* feature and sums absolute
    values, because the printed equations contradict the rest of the paper: Section 4.2 states
    that the contribution depends on "the alignment of signs between the weights and relative
    features", the abstract describes the decoupling the same way, and the caption of Figure 2
    defines relative features as the gap between sample and class-average features. Summing
    absolute values is also the only reading under which
    :math:`\\mathcal{E}_p + \\mathcal{E}_n` recovers the CARef error, as Section 4.3 asserts.

    Like :class:`CARef`, this detector only accepts pooled features of shape :math:`(N, D)`
    and inherits the predicted-label centroids and their fallback for unpredicted classes.

    .. note::
        The original publication defines :math:`\\text{Score}_{CADRef} = -s(x)`, so that
        in-distribution inputs obtain the larger value. This implementation returns
        :math:`s(x)` itself, following this library's convention that larger scores indicate
        outliers.

    .. note::
        :math:`\\bar{\\mathcal{S}}` is estimated on the training data during :meth:`fit`, so
        scores remain independent of the composition of the batch or of the evaluation set.
        Changing ``logit_score`` after fitting leaves :attr:`mean_logit_score` stale; refit
        before scoring again. :class:`pytorch_ood.utils.GridSearch` refits automatically.

    Example Code:

    .. code :: python

        model = WideResNet().eval()
        detector = CADRef(encoder=model.features, head=model.fc)
        detector.fit(train_loader)
        scores = detector(images)

    :see Paper:
        `ArXiv <https://arxiv.org/abs/2503.00325>`__

    :see Implementation:
        `GitHub <https://github.com/LingAndZero/CADRef>`__

    """

    requires_fit = True

    #: search space for :class:`pytorch_ood.utils.GridSearch`, covering the logit-based scores
    #: ablated in the publication. Changing this requires refitting, which the grid search does.
    hyperparameter_space = {
        "logit_score": [
            EnergyBased.score,
            _gen_score,
            MaxLogit.score,
            MaxSoftmax.score,
        ]
    }

    def __init__(
        self,
        encoder: Optional[Callable[[Tensor], Tensor]],
        head: Linear,
        logit_score: Callable[[Tensor], Tensor] = None,
    ) -> None:
        """
        :param encoder: model mapping inputs to penultimate-layer features. Can be ``None``
            when using ``fit_features(...)`` and ``predict_features(...)`` directly.
        :param head: the linear classification head of the model. Required, both for fitting
            and for scoring. Must expose a weight matrix of shape ``(num_classes,
            num_features)``, whose sign pattern defines the decoupling.
        :param logit_score: function mapping logits of shape ``(N, C)`` to outlier scores of
            shape ``(N,)``, following this library's convention that larger means more
            outlier. Its negation is used as the confidence :math:`\\mathcal{S}`, which the
            error scaling implicitly assumes to be positive. Default is
            :meth:`pytorch_ood.detector.EnergyBased.score`, the choice of the publication;
            :meth:`pytorch_ood.detector.MaxLogit.score` and
            :meth:`pytorch_ood.detector.MaxSoftmax.score` reproduce the other ablations, and
            :meth:`gen_score` the GEN one. Note that
            :meth:`pytorch_ood.detector.GEN.score` is *not* usable here, see :meth:`gen_score`.
        """
        super(CADRef, self).__init__(encoder=encoder, head=head)

        weight = getattr(head, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(
                "head must expose a weight matrix of shape (num_classes, num_features), "
                f"got {None if weight is None else tuple(weight.shape)}"
            )

        #: outlier score whose negation is used as the confidence for error scaling
        self.logit_score = logit_score or EnergyBased.score
        self.mean_logit_score: Optional[Tensor] = None  #: average confidence of the ID train set

    def __repr__(self):
        return f"CADRef(logit_score={getattr(self.logit_score, '__qualname__', self.logit_score)})"

    #: GEN-based outlier score reproducing the publication's GEN ablation
    gen_score = staticmethod(_gen_score)

    def _confidence(self, logits: Tensor) -> Tensor:
        """
        Confidence used for error scaling: larger for inliers, and assumed positive.
        """
        return -self.logit_score(logits)

    @torch.no_grad()
    def _fit_mean_confidence(self, z: Tensor, batch_size: int) -> Tensor:
        """
        Average logit-based confidence over the fitting features, chunked like
        :meth:`CARef._fit_class_means`. The per-sample confidences are kept on the CPU, where
        they occupy one float per sample, so a user-supplied callable may return them on any
        device.
        """
        device = self.device or z.device

        confidences = []
        for start in range(0, z.shape[0], batch_size):
            logits = self.head(z[start : start + batch_size].to(device))
            confidences.append(self._confidence(logits).detach().flatten().cpu())

        return torch.cat(confidences).mean().to(device)

    @torch.no_grad()
    def fit_features(
        self: Self, z: Tensor, y: Optional[Tensor] = None, batch_size: int = 4096
    ) -> Self:
        """
        Estimate the class-aware average features and the average logit-based confidence.
        Ignores OOD samples.

        :param z: features of the training data, shape ``(N, D)``. May live on a different
            device than the detector; chunks are moved as needed.
        :param y: corresponding class labels. When given, OOD samples are discarded. The
            labels are not used to form the centroids, see :class:`CARef`.
        :param batch_size: chunk size used while passing features through the head
        """
        z = self._prepare_fit_features(z, y)
        self.train_means = self._fit_class_means(z, batch_size)
        self.mean_logit_score = self._fit_mean_confidence(z, batch_size)
        return self

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Calculate outlier scores based on features.

        :param z: features as given by the model, shape ``(N, D)``
        :return: outlier scores, higher means more outlier
        """
        if self.train_means is None or self.mean_logit_score is None:
            raise RequiresFittingException()

        device = self.device or z.device
        z = self._as_features(z).to(device)

        logits = self.head(z)
        y_hat = logits.argmax(dim=1)

        relative = z - self.train_means[y_hat]
        # dimensions with zero weight do not contribute to the logit and are dropped from both
        # components, as in the reference implementation
        aligned = relative * self.head.weight[y_hat].sign()

        feature_norm = z.norm(p=1, dim=1).clamp(min=self._NORM_EPS)
        positive_error = aligned.clamp(min=0).sum(dim=1) / feature_norm
        negative_error = (-aligned).clamp(min=0).sum(dim=1) / feature_norm

        return positive_error / self._confidence(logits) + negative_error / self.mean_logit_score
