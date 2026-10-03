"""
Metrics measure how well a detector separates in-distribution (ID) from out-of-distribution
(OOD) samples. Outlier scores are larger for samples that are more likely OOD, and OOD samples
have labels :math:`< 0`. The metrics are computed in torch, on the device of your data, so
evaluation on the GPU does not need to copy the scores to the CPU.

For most evaluations, :class:`~pytorch_ood.metrics.OODMetrics` is all you need: feed it the
outlier scores and labels of each batch, and read all commonly reported metrics at the end.

.. code-block:: python

    from pytorch_ood.metrics import OODMetrics

    metrics = OODMetrics()
    for x, y in loader:
        metrics.update(detector(x), y)
    print(metrics.compute())  # {"AUROC": ..., "AUTC": ..., "AUPR-IN": ..., ...}

Every metric works this way (see :class:`~pytorch_ood.metrics.Metric`), so you can also
compute a single one, such as :class:`~pytorch_ood.metrics.AUROC`, or combine your own
selection in a :class:`~pytorch_ood.metrics.MetricCollection`. Metrics based on curves need
all scores at once, so they keep the scores and labels in memory until the end; with large
datasets or segmentation, pass ``device="cpu"`` to keep them off the GPU. If you already have
all scores, the functions in :mod:`pytorch_ood.metrics.functional` compute the metrics
directly.
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
