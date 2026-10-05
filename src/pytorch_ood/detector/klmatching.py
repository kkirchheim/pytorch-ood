"""

..  autoclass:: pytorch_ood.detector.KLMatching
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ParameterDict
from typing_extensions import Self

from ..api import (
    DetectorInfo,
    LogitsDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils import is_known

log = logging.getLogger()


class KLMatching(LogitsDetector):
    """
    Implements KL-Matching from the paper *Scaling Out-of-Distribution Detection for Real-World Settings*.

    For each class :math:`k`, a typical posterior distribution :math:`d_k` is estimated as the mean
    posterior of the samples of the fitted data (a validation set :math:`\\mathcal{X}_{\\text{val}}`) that the
    model predicts as class :math:`k`:

    .. math::
        d_k = \\mathbb{E}_{x' \\sim \\mathcal{X}_{\\text{val}}} [p(y \\vert x') \\mid \\arg\\max_y p(y \\vert x') = k]

    This requires no class labels. The outlier score of an input :math:`x` is the KL divergence of its
    posterior to the closest typical posterior:

    .. math::
        \\min_k D_{KL}[p(y \\vert x) \\Vert d_k]

    where the minimum runs over the classes that the model predicts for at least one sample of the fitted
    data.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Scaling Out-of-Distribution Detection for Real-World Settings",
            venue="ICML",
            year=2022,
            url="https://arxiv.org/abs/1911.11132",
            code=None,
        ),
        tasks={Task.CLASSIFICATION},
    )

    requires_fit = True

    def __init__(self, model: Optional[Module]):
        """
        :param model: neural network, is assumed to output logits. Can be ``None`` when
            using ``fit_logits(...)`` and ``predict_logits(...)`` directly.
        """
        super(KLMatching, self).__init__()
        self.model = model
        self.dists: ParameterDict = ParameterDict()  #: Typical posteriors per class

    def fit_logits(self, logits: Tensor, labels: Optional[Tensor] = None) -> Self:
        """
        Estimates the typical posterior of each class from the samples the model predicts as that
        class. The class labels are not needed; if given, they are only used to ignore OOD samples.

        :param logits: logits of shape :math:`N \\times C`
        :param labels: labels of shape :math:`N`; samples with labels :math:`< 0` are ignored
        :return: the fitted detector
        """
        device = self.device or logits.device
        logits = logits.to(device)
        if labels is not None:
            logits = logits[is_known(labels.to(device))]
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

        self.dists = ParameterDict()
        for k in predictions.unique():
            log.debug(f"Fitting class {k}")
            d_k = probabilities[predictions == k].mean(dim=0)
            self.dists[str(k.item())] = Parameter(d_k, requires_grad=False)

        return self

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits predicted by the model, shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`
        :raises RequiresFittingException: if the detector was not fitted
        """
        if len(self.dists) == 0:
            raise RequiresFittingException("KL-Matching has to be fitted on validation data.")
        p = logits.softmax(dim=1)
        return self._score_probabilities(p)

    def _score_probabilities(self, p: Tensor) -> Tensor:
        """
        Score already-computed posterior probabilities.

        :param p: probabilities predicted by the model, shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`
        """
        dists = torch.stack([d.to(p.device) for d in self.dists.values()])  # K x C
        # KL[p || d_k] = sum p log p - sum p log d_k for every input and fitted class, B x K.
        # 0 log 0 = 0; log 0 is replaced by the lowest finite value, so that probabilities of 0
        # contribute nothing and positive ones where d_k is 0 give a (near) infinite divergence.
        log_d = dists.log().nan_to_num(neginf=torch.finfo(dists.dtype).min)
        p_log_p = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
        kl = p_log_p.sum(dim=1, keepdim=True) - p @ log_d.T
        return kl.min(dim=1).values

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculates KL-Divergence between predicted posteriors and typical posteriors.

        :param x: input tensor, will be passed through model
        :return: outlier scores of shape :math:`B`
        """
        if len(self.dists) == 0:
            raise RequiresFittingException("KL-Matching has to be fitted on validation data.")

        if self.model is None:
            raise ModelNotSetException

        # we move the dict with the typical posteriors to the same device as the input
        # this might be not desirable in some cases, but avoids errors
        device = x.device
        self.dists.to(device)

        logits = self.model(x)
        return self.predict_logits(logits)
