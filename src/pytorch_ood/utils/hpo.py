"""

..  autoclass:: pytorch_ood.utils.GridSearch
    :members:

"""

import logging
import math
from itertools import product
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..api import Detector, FeaturesDetector, GradientDetector, LogitsDetector
from .metrics import OODMetrics
from .utils import TensorBuffer, extract_features

__all__ = ["GridSearch"]

log = logging.getLogger(__name__)


@torch.no_grad()
def _extract_all(data_loader: DataLoader, producer, device: str):
    """
    Extract representations for *all* samples in a loader, keeping both ID and OOD.

    Unlike :func:`pytorch_ood.utils.extract_features`, OOD samples are retained,
    so the result can be scored against ID/OOD labels.
    """
    buffer = TensorBuffer(device="cpu")
    for x, y in data_loader:
        z = producer(x.to(device))
        z = z.view(z.shape[0], -1)
        buffer.append("z", z)
        buffer.append("y", y)
    return buffer.get("z"), buffer.get("y")


class GridSearch:
    """
    Grid search over a detector's hyperparameters, replicating the automatic
    parameter search (APS) protocol used by OpenOOD: every combination in the
    search space is evaluated on a held-out validation set, and the combination
    that optimizes the target metric is selected.

    The validation loader must contain both in-distribution (label ``>= 0``) and
    out-of-distribution (label ``< 0``) samples.

    For detectors that expose pooled features (``encoder``) or logits (``model``),
    the model outputs are extracted **once** and reused across all candidates, so
    each candidate only re-runs the cheap fit/score step on cached tensors. For
    other detectors (e.g. gradient-based), the full forward pass is repeated per
    candidate.

    .. note::
        This **mutates the detector**: on return it is fitted (if required) and
        configured with the best hyperparameters. For detectors evaluated via the
        non-cached path (e.g. those operating on raw inputs), put the underlying
        model in ``eval()`` mode so that dropout/batch-norm noise does not confound
        the comparison across candidates. For segmentation, pass a metric such as
        ``OODMetrics(mode="segmentation")``.

    .. code :: python

        detector = ASH(backbone=..., head=...)
        search = GridSearch(detector, fit_loader=train_loader, val_loader=val_loader)
        best = search.run()             # also leaves `detector` set to the best params
        print(best, search.best_score_)

    :param detector: detector to tune; must define a non-empty ``hyperparameter_space``
        unless one is passed explicitly
    :param fit_loader: data used to (re-)fit the detector per candidate. Only required
        if the detector requires fitting.
    :param val_loader: validation data containing both ID and OOD samples
    :param hyperparameter_space: overrides the detector's ``hyperparameter_space``
    :param metric: metric object with ``update(scores, y)`` / ``compute() -> dict``
        and ``reset()``. Defaults to :class:`pytorch_ood.utils.OODMetrics`.
    :param metric_name: key to read from the metric's ``compute()`` dict. Default ``"AUROC"``.
    :param higher_is_better: whether the metric should be maximized. Default ``True``.
    :param device: device used for extraction and scoring
    """

    def __init__(
        self,
        detector: Detector,
        fit_loader: Optional[DataLoader],
        val_loader: DataLoader,
        hyperparameter_space: Optional[Dict[str, List]] = None,
        metric=None,
        metric_name: str = "AUROC",
        higher_is_better: bool = True,
        device: str = "cpu",
    ):
        self.detector = detector
        self.fit_loader = fit_loader
        self.val_loader = val_loader
        self.space = hyperparameter_space or detector.hyperparameter_space
        if not self.space:
            raise ValueError(
                f"{type(detector).__name__} has an empty hyperparameter_space; "
                "nothing to search. Pass hyperparameter_space explicitly."
            )
        for name, values in self.space.items():
            if not values:
                raise ValueError(f"Search space for hyperparameter '{name}' is empty.")
        self.metric = metric if metric is not None else OODMetrics()
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.device = device

        self.best_params_: Optional[Dict] = None  #: best hyperparameters found
        self.best_score_: Optional[float] = None  #: best validation score
        self.results_: List[Dict] = []  #: per-candidate ``{params, score}`` records

    def _candidates(self):
        names = list(self.space.keys())
        for values in product(*(self.space[name] for name in names)):
            yield dict(zip(names, values))

    def _read_metric(self) -> float:
        result = self.metric.compute()
        if self.metric_name not in result:
            raise ValueError(
                f"metric_name '{self.metric_name}' is not produced by the metric; "
                f"available metrics: {sorted(result)}"
            )
        return float(result[self.metric_name])

    def _score(self, scores: Tensor, y: Tensor) -> float:
        self.metric.reset()
        self.metric.update(scores, y)
        return self._read_metric()

    def run(self) -> Dict:
        """
        Run the grid search.

        :return: the best hyperparameter combination. As a side effect, ``detector``
            is left fitted (if required) and configured with these values.
        :raise ValueError: if a detector that requires fitting is given no
            ``fit_loader``, or if no candidate produced a finite score.
        """
        self.detector.to(self.device)

        # Make the resolved search space authoritative so set_hyperparameters accepts its
        # keys, whether the space came from the detector or the override argument.
        self.detector.hyperparameter_space = self.space

        if self.detector.requires_fit and self.fit_loader is None:
            raise ValueError("fit_loader is required to tune a detector that requires fitting")

        # Decide whether model outputs can be extracted once and cached.
        cached_features = (
            isinstance(self.detector, FeaturesDetector)
            and not isinstance(self.detector, GradientDetector)
            and getattr(self.detector, "encoder", None) is not None
        )
        cached_logits = (
            isinstance(self.detector, LogitsDetector)
            and getattr(self.detector, "model", None) is not None
        )

        if cached_features:
            producer = self.detector.encoder
        elif cached_logits:
            producer = self.detector.model
        else:
            producer = None

        z_train = y_train = z_val = y_val = None
        if producer is not None:
            if self.detector.requires_fit:
                z_train, y_train = extract_features(self.fit_loader, producer, self.device)
            z_val, y_val = _extract_all(self.val_loader, producer, self.device)

        self.results_ = []
        for params in self._candidates():
            self.detector.set_hyperparameters(**params)
            self._fit(cached_features, cached_logits, z_train, y_train)
            score = self._predict_score(cached_features, cached_logits, z_val, y_val)
            self.results_.append({"params": params, "score": score})

            # each candidate's fit/predict can allocate large transient CUDA tensors
            # (e.g. ViM's principal-subspace projection); release them back to the
            # allocator before the next candidate to avoid fragmentation-driven OOMs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # never select a non-finite score (e.g. NaN from a degenerate candidate)
            if not math.isfinite(score):
                continue

            if (
                self.best_score_ is None
                or (self.higher_is_better and score > self.best_score_)
                or (not self.higher_is_better and score < self.best_score_)
            ):
                self.best_score_ = score
                self.best_params_ = params

        if self.best_params_ is None:
            raise ValueError(
                "No hyperparameter combination produced a finite score; cannot select a best."
            )

        # Leave the detector consistently configured AND fitted for the best params
        # (the last candidate evaluated is generally not the best one).
        self.detector.set_hyperparameters(**self.best_params_)
        self._fit(cached_features, cached_logits, z_train, y_train)
        return self.best_params_

    def _fit(self, cached_features, cached_logits, z_train, y_train) -> None:
        if not self.detector.requires_fit:
            return
        if cached_features:
            self.detector.fit_features(z_train, y_train)
        elif cached_logits:
            self.detector.fit_logits(z_train, y_train)
        else:
            self.detector.fit(self.fit_loader)

    def _predict_score(self, cached_features, cached_logits, z_val, y_val) -> float:
        if cached_features:
            return self._score_or_nan(self.detector.predict_features(z_val), y_val)

        if cached_logits:
            return self._score_or_nan(self.detector.predict_logits(z_val), y_val)

        # Fallback: no caching possible, run the full pipeline per candidate.
        self.metric.reset()
        for x, y in self.val_loader:
            scores = self.detector.predict(x.to(self.device))
            # Non-finite scores must not be scored: some metrics (e.g. AUROC via
            # torchmetrics) map all-NaN scores to a spurious perfect value.
            if not torch.isfinite(scores).all():
                return float("nan")
            self.metric.update(scores, y.to(scores.device))
        return self._read_metric()

    def _score_or_nan(self, scores: Tensor, y: Tensor) -> float:
        if not torch.isfinite(scores).all():
            return float("nan")
        return self._score(scores, y)
