"""
Metrics for anomaly segmentation, where each pixel is a sample.
"""

import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Self

from .base import Device, Metric, MetricCollection, StreamingMetric
from .ood import AUPR, AUROC, AUTC, FPRAtTPR

__all__ = ["PerImage", "OODSegmentationMetrics", "OODPerImageSegmentationMetrics"]


def _ood_metrics(fpr_at: float):
    return [AUROC(), AUTC(), AUPR("id"), AUPR("ood"), FPRAtTPR(fpr_at)]


class OODSegmentationMetrics(MetricCollection):
    """
    The metrics of :class:`~pytorch_ood.metrics.OODMetrics` for anomaly segmentation, computed
    over the pixels of all images together, as in the SegmentMeIfYouCan and Fishyscapes
    benchmarks. Images without OOD pixels therefore count as well. :meth:`compute` returns a
    dictionary with the keys ``AUROC``, ``AUTC``, ``AUPR-IN``, ``AUPR-OUT``, and ``FPR95TPR``.

    The scores and labels of all pixels are stored until :meth:`compute` is called. For large
    datasets, pass ``device="cpu"`` to keep them in main memory instead of on the GPU. To
    compute the metrics for each image separately, use
    :class:`~pytorch_ood.metrics.OODPerImageSegmentationMetrics`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import OODSegmentationMetrics

        metrics = OODSegmentationMetrics(device="cpu", void_label=255)
        for x, y in loader:
            metrics.update(detector(x), y)  # score maps and label masks of shape B x H x W
        print(metrics.compute())
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
        :param void_label: label of pixels to ignore, e.g., unlabeled pixels
        :raises ValueError: if ``fpr_at`` is not in :math:`[0, 1]` or ``void_label`` is negative
        """
        super().__init__(_ood_metrics(fpr_at), device=device, void_label=void_label)

    def update(self, scores: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch.

        :param scores: outlier scores of any shape, e.g., :math:`B \\times H \\times W`
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD pixels
        :return: self
        :raises ValueError: if the shapes differ
        """
        return super().update(scores=scores, labels=labels)


class PerImage(StreamingMetric):
    """
    Computes a metric for each image separately and reports the mean over the images. The first
    dimension of the inputs indexes the images, e.g., :math:`B \\times H \\times W` for score
    maps and label masks.

    Images for which the metric is undefined are skipped: images without ID or without OOD
    pixels for metrics based on curves, such as :class:`~pytorch_ood.metrics.AUROC`, and images
    without ID pixels for :class:`~pytorch_ood.metrics.Accuracy` (see
    :attr:`~pytorch_ood.metrics.Metric.needs_id` and
    :attr:`~pytorch_ood.metrics.Metric.needs_ood`), as well as images whose pixels all have
    ``void_label``. :meth:`compute` warns about skipped images and raises if all images were
    skipped. Only the sums of the results are stored, so the memory does not grow with the
    number of images.

    A per-image metric can not be part of a :class:`~pytorch_ood.metrics.MetricCollection`.
    To compute several metrics per image, wrap the collection instead.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import AUROC, PerImage

        metric = PerImage(AUROC())
        for x, y in loader:
            metric.update(detector(x), y)
        print(metric.compute())  # {"AUROC": ...}, the mean over the images
    """

    _per_image = True

    def __init__(
        self,
        metric: Metric,
        *,
        device: Optional[Device] = None,
        void_label: Optional[int] = None,
    ):
        """
        :param metric: the metric to compute for each image, e.g., a
            :class:`~pytorch_ood.metrics.MetricCollection`. It must take ``labels`` as an
            input, and must not set ``device`` or ``void_label`` itself.
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``metric`` does not take labels, is itself a per-image metric,
            or sets its own ``device`` or ``void_label``
        """
        super().__init__(device=device, void_label=void_label)
        if metric._per_image:
            raise ValueError(f"{type(metric).__name__} already computes metrics per image")
        if "labels" not in metric.inputs:
            raise ValueError(f"{type(metric).__name__} does not take labels")
        if metric.device is not None or metric.void_label is not None:
            raise ValueError(
                f"{type(metric).__name__} sets its own device or void_label; set them on "
                f"PerImage instead"
            )
        self.metric = metric
        self.inputs = metric.inputs
        self._reset()

    @property
    def keys(self) -> Tuple[str, ...]:
        return self.metric.keys

    def update(self, *args: Tensor, **kwargs: Tensor) -> Self:
        """
        Adds a batch of images.

        :return: self
        :raises TypeError: if an input is not a tensor, is not one of
            :attr:`~pytorch_ood.metrics.Metric.inputs`, or is given twice
        :raises ValueError: if inputs are missing, have different shapes, or have no image
            dimension
        """
        inputs = self._bind(args, kwargs)
        missing = [name for name in self.inputs if inputs.get(name) is None]
        if missing:
            raise ValueError(f"{type(self).__name__}.update is missing the inputs {missing}")
        for name, value in inputs.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Input {name!r} must be a tensor, got {type(value).__name__}")
        shapes = {tuple(value.shape) for value in inputs.values()}
        if len(shapes) > 1:
            raise ValueError(
                "Inputs must have the same shape, got "
                + ", ".join(f"{name}: {tuple(value.shape)}" for name, value in inputs.items())
            )
        shape = shapes.pop()
        if len(shape) == 0:
            raise ValueError("Inputs must have an image dimension")
        # each image is flattened, filtered, and moved separately
        for i in range(shape[0]):
            image = self._prepare({name: value[i] for name, value in inputs.items()})
            self._update(image)
        return self

    def _update(self, inputs: Optional[Dict[str, Tensor]]) -> None:
        self._images += 1
        if inputs is None:
            # every pixel is void
            return
        if self.metric.needs_id or self.metric.needs_ood:
            n_ood = int((inputs["labels"] < 0).sum())
            if self.metric.needs_ood and n_ood == 0:
                return
            if self.metric.needs_id and n_ood == inputs["labels"].numel():
                return
        self.metric._update(inputs)
        results = self.metric._compute()
        # the inner metric only ever holds one image
        self.metric._reset()
        for key, value in results.items():
            self._sums[key] = self._sums.get(key, 0) + value.double()
        self._counted += 1

    def _compute(self) -> Dict[str, Tensor]:
        if self._images == 0:
            raise ValueError(f"{type(self).__name__} was given no data")
        reasons = [
            f"without {name} pixels"
            for name, needs in (("ID", self.metric.needs_id), ("OOD", self.metric.needs_ood))
            if needs
        ]
        reasons.append("with only void pixels")
        reason = ", ".join(reasons[:-1]) + " or " + reasons[-1] if len(reasons) > 1 else reasons[0]
        if self._counted == 0:
            raise ValueError(f"All {self._images} images were skipped: images {reason}")
        skipped = self._images - self._counted
        if skipped:
            warnings.warn(f"Skipped {skipped} of {self._images} images {reason}")
        return {key: value / self._counted for key, value in self._sums.items()}

    def _reset(self) -> None:
        self._sums: Dict[str, Tensor] = {}
        self._images = 0
        self._counted = 0


class OODPerImageSegmentationMetrics(PerImage):
    """
    The metrics of :class:`~pytorch_ood.metrics.OODMetrics` for anomaly segmentation, computed
    for each image separately and averaged over the images. :meth:`compute` returns a
    dictionary with the keys ``AUROC``, ``AUTC``, ``AUPR-IN``, ``AUPR-OUT``, and ``FPR95TPR``.

    Images without ID or without OOD pixels are skipped, see
    :class:`~pytorch_ood.metrics.PerImage`. To compute the metrics over the pixels of all
    images together, use :class:`~pytorch_ood.metrics.OODSegmentationMetrics`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import OODPerImageSegmentationMetrics

        metrics = OODPerImageSegmentationMetrics()
        for x, y in loader:
            metrics.update(detector(x), y)  # score maps and label masks of shape B x H x W
        print(metrics.compute())
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
        :param void_label: label of pixels to ignore, e.g., unlabeled pixels
        :raises ValueError: if ``fpr_at`` is not in :math:`[0, 1]` or ``void_label`` is negative
        """
        super().__init__(
            MetricCollection(_ood_metrics(fpr_at)), device=device, void_label=void_label
        )

    def update(self, scores: Tensor, labels: Tensor) -> Self:
        """
        Adds a batch of images.

        :param scores: outlier scores of shape :math:`B \\times \\ldots`, e.g.,
            :math:`B \\times H \\times W`
        :param labels: labels of the same shape; labels :math:`< 0` mark OOD pixels
        :return: self
        :raises ValueError: if the shapes differ, or the inputs have no image dimension
        """
        return super().update(scores=scores, labels=labels)
