"""
Metrics for OOD detection, computed in torch on the device of the data.

Each metric implements :class:`~pytorch_ood.metrics.Metric`: :meth:`~pytorch_ood.metrics.Metric.update` adds a batch,
:meth:`~pytorch_ood.metrics.Metric.compute` returns the results as a dictionary of floats, and
:meth:`~pytorch_ood.metrics.Metric.reset` starts over. :class:`~pytorch_ood.metrics.StreamingMetric` subclasses reduce each batch to a
fixed-size state; :class:`~pytorch_ood.metrics.BufferedMetric` subclasses, such as the areas under curves, store
their inputs until :meth:`~pytorch_ood.metrics.Metric.compute`. :class:`~pytorch_ood.metrics.MetricCollection` computes several metrics
and stores each input only once. :class:`~pytorch_ood.metrics.OODMetrics` is the collection of the metrics commonly
reported for OOD detection.

The functions in :mod:`pytorch_ood.metrics.functional` compute the metrics from complete tensors.
"""

from . import functional
from .base import BufferedMetric, Metric, MetricCollection, StreamingMetric
from .functional import aurra, calc_openness, calibration_error, oscr_score
from .ood import AUPR, AUROC, AUTC, Accuracy, FPRAtTPR, OODMetrics

__all__ = [
    "Metric",
    "StreamingMetric",
    "BufferedMetric",
    "MetricCollection",
    "AUROC",
    "AUPR",
    "FPRAtTPR",
    "AUTC",
    "Accuracy",
    "OODMetrics",
    "oscr_score",
    "calibration_error",
    "aurra",
    "calc_openness",
    "functional",
]
