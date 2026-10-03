"""
Metrics for OOD detection and closed-set classification.
"""

import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Self

from ..utils.utils import _check_fraction
from . import functional as F
from .base import BufferedMetric, Device, MetricCollection, StreamingMetric, _Data

__all__ = ["AUROC", "AUPR", "FPRAtTPR", "AUTC", "Accuracy", "OODMetrics"]


class AUROC(BufferedMetric):
    """
    Area under the receiver operating characteristic curve, with OOD samples (labels
    :math:`< 0`) as the positive class: the probability that a random OOD sample gets a higher
    outlier score than a random ID sample, where ties count half. See
    :func:`~pytorch_ood.metrics.functional.auroc`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import AUROC

        metric = AUROC()
        for x, y in loader:
            metric.update(detector(x), y)
        print(metric.compute())  # {"AUROC": ...}
    """

    inputs = ("scores", "labels")

    def __init__(self, *, device: Optional[Device] = None, void_label: Optional[int] = None):
        """
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``void_label`` is negative
        """
        super().__init__(device=device, void_label=void_label)

    def update(self, scores: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch.

        :param scores: outlier scores of any shape; larger means more likely OOD
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
        :return: self
        :raises ValueError: if the shapes differ
        """
        return super().update(scores, labels)

    def compute(self) -> Dict[str, float]:
        """
        Computes the AUROC from all batches added so far.

        :return: ``{"AUROC": value}``, with a value in :math:`[0, 1]`. Higher is better; a
            detector that guesses gets 0.5.
        :raises ValueError: if no data was given, there are no ID or no OOD samples, or the
            scores contain NaN
        """
        return super().compute()

    @property
    def keys(self) -> Tuple[str, ...]:
        return ("AUROC",)

    def _compute_from(self, data: _Data) -> Dict[str, Tensor]:
        fpr, tpr = F._roc(data.counts)
        return {"AUROC": F._area(fpr, tpr)}


class AUPR(BufferedMetric):
    """
    Area under the precision-recall curve, by the trapezoidal rule. With ``positive="ood"``
    (AUPR-OUT), OOD samples (labels :math:`< 0`) are the positive class; with
    ``positive="id"`` (AUPR-IN), ID samples are, and lower outlier scores mean more likely
    positive. See :func:`~pytorch_ood.metrics.functional.aupr`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import AUPR

        metric = AUPR(positive="id")
        for x, y in loader:
            metric.update(detector(x), y)
        print(metric.compute())  # {"AUPR-IN": ...}
    """

    inputs = ("scores", "labels")

    def __init__(
        self,
        positive: str = "ood",
        *,
        device: Optional[Device] = None,
        void_label: Optional[int] = None,
    ):
        """
        :param positive: the positive class, ``"ood"`` or ``"id"``
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``positive`` is invalid or ``void_label`` is negative
        """
        if positive not in ("ood", "id"):
            raise ValueError(f"positive must be 'ood' or 'id', got {positive!r}")
        super().__init__(device=device, void_label=void_label)
        self.positive = positive

    def update(self, scores: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch.

        :param scores: outlier scores of any shape; larger means more likely OOD
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
        :return: self
        :raises ValueError: if the shapes differ
        """
        return super().update(scores, labels)

    def compute(self) -> Dict[str, float]:
        """
        Computes the AUPR from all batches added so far.

        :return: ``{"AUPR-OUT": value}`` for ``positive="ood"`` and ``{"AUPR-IN": value}`` for
            ``positive="id"``, with a value in :math:`[0, 1]`. Higher is better; a detector
            that guesses gets the fraction of positive samples.
        :raises ValueError: if no data was given, there are no ID or no OOD samples, or the
            scores contain NaN
        """
        return super().compute()

    @property
    def keys(self) -> Tuple[str, ...]:
        return ("AUPR-OUT",) if self.positive == "ood" else ("AUPR-IN",)

    def _compute_from(self, data: _Data) -> Dict[str, Tensor]:
        return {self.keys[0]: F._aupr(data.counts, self.positive)}


class FPRAtTPR(BufferedMetric):
    """
    False positive rate at the threshold where the true positive rate first reaches ``tpr``,
    with OOD samples as the positive class: the fraction of ID samples that are flagged as OOD
    when the threshold is set so that, e.g., 95% of the OOD samples are detected. See
    :func:`~pytorch_ood.metrics.functional.fpr_at_tpr`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import FPRAtTPR

        metric = FPRAtTPR(tpr=0.95)
        for x, y in loader:
            metric.update(detector(x), y)
        print(metric.compute())  # {"FPR95TPR": ...}
    """

    inputs = ("scores", "labels")

    def __init__(
        self,
        tpr: float = 0.95,
        *,
        device: Optional[Device] = None,
        void_label: Optional[int] = None,
    ):
        """
        :param tpr: true positive rate, a fraction in :math:`[0, 1]`
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``tpr`` is not in :math:`[0, 1]` or ``void_label`` is negative
        """
        super().__init__(device=device, void_label=void_label)
        self.tpr = _check_fraction("tpr", tpr)

    def update(self, scores: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch.

        :param scores: outlier scores of any shape; larger means more likely OOD
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
        :return: self
        :raises ValueError: if the shapes differ
        """
        return super().update(scores, labels)

    def compute(self) -> Dict[str, float]:
        """
        Computes the false positive rate from all batches added so far.

        :return: a dictionary with one entry, named after ``tpr`` in percent: ``{"FPR95TPR":
            value}`` for ``tpr=0.95``, ``{"FPR90TPR": value}`` for ``tpr=0.9``. The value is in
            :math:`[0, 1]`; lower is better.
        :raises ValueError: if no data was given, there are no ID or no OOD samples, or the
            scores contain NaN
        """
        return super().compute()

    @property
    def keys(self) -> Tuple[str, ...]:
        return (f"FPR{self.tpr * 100:g}TPR",)

    def _compute_from(self, data: _Data) -> Dict[str, Tensor]:
        return {self.keys[0]: F._fpr_at_tpr(data.counts, self.tpr)}


class AUTC(BufferedMetric):
    """
    Area under the threshold curve: the false positive and the false negative rate, averaged
    over all thresholds, after the outlier scores are scaled to :math:`[0, 1]`. Unlike AUROC,
    it also reflects how far apart the scores of ID and OOD samples are. Lower is better; 0
    means that all ID samples get the lowest and all OOD samples the highest score. See
    :func:`~pytorch_ood.metrics.functional.autc`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import AUTC

        metric = AUTC()
        for x, y in loader:
            metric.update(detector(x), y)
        print(metric.compute())  # {"AUTC": ...}

    AUTC is undefined for constant or infinite scores. In these cases, the result is NaN and a
    warning is issued, so that collections such as :class:`~pytorch_ood.metrics.OODMetrics`
    still report the other metrics.
    """

    inputs = ("scores", "labels")

    def __init__(self, *, device: Optional[Device] = None, void_label: Optional[int] = None):
        """
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``void_label`` is negative
        """
        super().__init__(device=device, void_label=void_label)

    def update(self, scores: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch.

        :param scores: outlier scores of any shape; larger means more likely OOD
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
        :return: self
        :raises ValueError: if the shapes differ
        """
        return super().update(scores, labels)

    def compute(self) -> Dict[str, float]:
        """
        Computes the AUTC from all batches added so far.

        :return: ``{"AUTC": value}``, with a value in :math:`[0, 1]`; lower is better. The value
            is NaN, with a warning, if the scores are constant or contain infinite values.
        :raises ValueError: if no data was given, there are no ID or no OOD samples, or the
            scores contain NaN
        """
        return super().compute()

    @property
    def keys(self) -> Tuple[str, ...]:
        return ("AUTC",)

    def _compute_from(self, data: _Data) -> Dict[str, Tensor]:
        scores = data["scores"]
        # errors in the data raise as for the other metrics; only the undefined cases are NaN
        F._check_scores(scores)
        n_ood = int((data["labels"] < 0).sum())
        if n_ood == 0 or n_ood == scores.numel():
            raise ValueError("AUTC requires both ID and OOD samples")
        if not bool(torch.isfinite(scores).all()) or bool(scores.min() == scores.max()):
            warnings.warn("AUTC is undefined for constant or infinite scores, returning NaN")
            return {"AUTC": torch.tensor(float("nan"), dtype=torch.float64)}
        return {"AUTC": F.autc(scores, data["labels"])}


class Accuracy(StreamingMetric):
    """
    Closed-set accuracy: the fraction of ID samples (labels :math:`\\geq 0`) whose predicted
    class equals the label. OOD samples are ignored. Only counts are stored.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import Accuracy

        metric = Accuracy()
        for x, y in loader:
            metric.update(model(x).argmax(dim=1), y)
        print(metric.compute())  # {"ACC": ...}
    """

    inputs = ("predictions", "labels")

    def __init__(self, *, device: Optional[Device] = None, void_label: Optional[int] = None):
        """
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``void_label`` is negative
        """
        super().__init__(device=device, void_label=void_label)
        self._reset()

    def update(self, predictions: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch.

        :param predictions: predicted class indices of any shape
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples, which
            are ignored
        :return: self
        :raises ValueError: if the shapes differ
        """
        return super().update(predictions, labels)

    def compute(self) -> Dict[str, float]:
        """
        Computes the accuracy from all batches added so far.

        :return: ``{"ACC": value}``, with a value in :math:`[0, 1]`; higher is better
        :raises ValueError: if no data was given or there are no ID samples
        """
        return super().compute()

    @property
    def keys(self) -> Tuple[str, ...]:
        return ("ACC",)

    def _update(self, inputs: Dict[str, Tensor]) -> None:
        known = inputs["labels"] >= 0
        # no boolean indexing, and the counts stay on the device of the data, so updates do
        # not synchronize with the host
        correct = ((inputs["predictions"] == inputs["labels"]) & known).sum()
        if self._correct is None:
            self._correct = torch.zeros((), dtype=torch.long, device=correct.device)
            self._total = torch.zeros((), dtype=torch.long, device=correct.device)
        self._correct += correct
        self._total += known.sum()

    def _compute(self) -> Dict[str, Tensor]:
        if self._total is None or int(self._total) == 0:
            raise ValueError("Accuracy requires ID samples (labels >= 0)")
        return {"ACC": F._float64(self._correct) / F._float64(self._total)}

    def _reset(self) -> None:
        self._correct: Optional[Tensor] = None
        self._total: Optional[Tensor] = None


class OODMetrics(MetricCollection):
    """
    The metrics commonly reported for OOD detection, computed from outlier scores and labels.
    OOD samples have labels :math:`< 0`; they are the positive class, and larger outlier scores
    mean more likely OOD. :meth:`~pytorch_ood.metrics.Metric.compute` returns a dictionary with the keys

    - ``AUROC``: area under the ROC curve, see :class:`~pytorch_ood.metrics.AUROC`
    - ``AUTC``: area under the threshold curve, lower is better, see
      :class:`~pytorch_ood.metrics.AUTC`
    - ``AUPR-IN``: area under the precision-recall curve with ID samples as the positive class,
      see :class:`~pytorch_ood.metrics.AUPR`
    - ``AUPR-OUT``: area under the precision-recall curve with OOD samples as the positive class
    - ``FPR95TPR``: false positive rate at the threshold where the true positive rate reaches
      95%, see :class:`~pytorch_ood.metrics.FPRAtTPR`. With ``fpr_at=0.9``, the key is
      ``FPR90TPR``.
    - ``ACC``: closed-set accuracy on the ID samples, see :class:`~pytorch_ood.metrics.Accuracy`,
      if predicted class indices are passed to :meth:`~pytorch_ood.metrics.Metric.update`

    Inputs of any shape are flattened, so each entry counts as a sample. The scores and labels
    are stored until :meth:`~pytorch_ood.metrics.Metric.compute` is called, on the device of
    the first input unless ``device`` is given; the accuracy only stores counts.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import OODMetrics

        metrics = OODMetrics()
        for x, y in loader:
            metrics.update(detector(x), y)
        print(metrics.compute())

    Passing predicted class indices additionally reports the closed-set accuracy:

    .. code-block:: python

        metrics.update(detector(x), y, model(x).argmax(dim=1))
    """

    def __init__(
        self,
        *,
        fpr_at: float = 0.95,
        device: Optional[Device] = None,
        void_label: Optional[int] = None,
    ):
        """
        :param fpr_at: true positive rate at which the false positive rate is reported, a
            fraction in :math:`[0, 1]`
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: label of entries to ignore, e.g., unlabeled pixels
        :raises ValueError: if ``fpr_at`` is not in :math:`[0, 1]` or ``void_label`` is negative
        """
        accuracy = Accuracy()
        super().__init__(
            [AUROC(), AUTC(), AUPR("id"), AUPR("ood"), FPRAtTPR(fpr_at), accuracy],
            device=device,
            void_label=void_label,
        )
        self._optional = (accuracy,)

    def compute(self) -> Dict[str, float]:
        """
        Computes all metrics from the batches added so far.

        :return: dictionary with the entries listed above
        :raises ValueError: if no data was given, there are no ID or no OOD samples, or the
            scores contain NaN
        """
        return super().compute()

    def update(self, scores: Tensor, labels: Tensor, predictions: Optional[Tensor] = None) -> Self:
        """
        Adds a batch.

        :param scores: outlier scores, larger means more likely OOD
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
        :param predictions: predicted class indices of the same shape. If given, they must be
            given in all calls, and :meth:`~pytorch_ood.metrics.Metric.compute` additionally reports the accuracy.
        :return: self
        :raises ValueError: if the shapes differ, or ``predictions`` are given in some calls only
        """
        return super().update(scores=scores, labels=labels, predictions=predictions)
