"""
Base classes of the metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Self

from .functional import _Counts

__all__ = ["Metric", "StreamingMetric", "BufferedMetric", "MetricCollection"]

Device = Union[str, torch.device]

# inputs holding class labels, stored as int32: enough for any number of classes, and the
# negative labels of OOD samples
_LABEL_INPUTS = ("labels", "predictions")


class _Data:
    """
    The complete inputs of buffered metrics, with the curve counts computed at most once and
    shared by all metrics that need them.
    """

    def __init__(self, tensors: Dict[str, Tensor]):
        self.tensors = tensors
        self._counts: Optional[_Counts] = None

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    @property
    def counts(self) -> _Counts:
        if self._counts is None:
            self._counts = _Counts(self.tensors["scores"], self.tensors["labels"] < 0)
        return self._counts


class Metric(ABC):
    """
    Interface of all metrics. You feed a metric batch by batch with
    :meth:`~pytorch_ood.metrics.Metric.update`, read the results with
    :meth:`~pytorch_ood.metrics.Metric.compute`, and clear it with
    :meth:`~pytorch_ood.metrics.Metric.reset` to evaluate the next model or dataset.

    The inputs a metric takes are named in :attr:`~pytorch_ood.metrics.Metric.inputs`, e.g.,
    ``("scores", "labels")``. All inputs of one call must have the same shape. They are
    flattened, so each entry counts as a sample: for segmentation, you can pass score maps and
    label masks as they are, and every pixel is a sample. Entries whose label equals
    ``void_label`` are ignored.

    **Devices.** If ``device`` is given, all inputs are moved there. Otherwise, the metric
    stays on the device of the first input it is given (after construction or
    :meth:`~pytorch_ood.metrics.Metric.reset`), and moves later inputs there. The scores can,
    e.g., be on the GPU while the labels are on the CPU. Metrics that store their inputs keep
    them on this device until :meth:`~pytorch_ood.metrics.Metric.compute`, which can use a lot
    of GPU memory for large datasets or segmentation; pass ``device="cpu"`` to keep them in
    main memory instead.

    Labels and predictions are stored as 32-bit integers, so class indices must be smaller
    than :math:`2^{31}`.
    """

    #: names of the inputs of :meth:`~pytorch_ood.metrics.Metric.update`, in order
    inputs: Tuple[str, ...] = ()

    def __init__(self, *, device: Optional[Device] = None, void_label: Optional[int] = None):
        """
        :param device: device for the state and the computations. If ``None``, the device of the
            first input of the first call of :meth:`~pytorch_ood.metrics.Metric.update` is used.
        :param void_label: label of entries to ignore, e.g., unlabeled pixels
        :raises ValueError: if ``void_label`` is negative, which would mark OOD samples
        """
        if void_label is not None and void_label < 0:
            raise ValueError(f"void_label must not be negative (OOD labels), got {void_label}")
        self.device = torch.device(device) if device is not None else None
        self.void_label = void_label
        self._device: Optional[torch.device] = self.device

    @property
    @abstractmethod
    def keys(self) -> Tuple[str, ...]:
        """
        Keys of the dictionary returned by :meth:`~pytorch_ood.metrics.Metric.compute`.
        """

    def update(self, *args: Tensor, **kwargs: Tensor) -> Self:
        """
        Adds a batch. The inputs are named by :attr:`~pytorch_ood.metrics.Metric.inputs` and can
        be given by position or by name.

        :return: self
        :raises TypeError: if an input is not a tensor, is not one of
            :attr:`~pytorch_ood.metrics.Metric.inputs`, or is given twice
        :raises ValueError: if inputs are missing or have different shapes
        """
        inputs = self._bind(args, kwargs)
        missing = [name for name in self.inputs if inputs.get(name) is None]
        if missing:
            raise ValueError(f"{type(self).__name__}.update is missing the inputs {missing}")
        prepared = self._prepare(inputs)
        if prepared is not None:
            self._update(prepared)
        return self

    def compute(self) -> Dict[str, float]:
        """
        :return: dictionary that maps :attr:`~pytorch_ood.metrics.Metric.keys` to the values of the metric
        :raises ValueError: if no data was given, or the metric is undefined for the data, e.g.,
            curves without ID or without OOD samples, or scores that contain NaN
        """
        return {key: float(value) for key, value in self._compute().items()}

    def reset(self) -> Self:
        """
        Removes all data. The next call of :meth:`~pytorch_ood.metrics.Metric.update` fixes the device again, unless
        ``device`` was given.

        :return: self
        """
        self._device = self.device
        self._reset()
        return self

    @abstractmethod
    def _update(self, inputs: Dict[str, Tensor]) -> None:
        """Adds prepared (flattened, filtered, moved) inputs."""

    @abstractmethod
    def _compute(self) -> Dict[str, Tensor]:
        """Computes the results."""

    @abstractmethod
    def _reset(self) -> None:
        """Removes all data."""

    def _bind(self, args: Sequence[Tensor], kwargs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if len(args) > len(self.inputs):
            raise TypeError(
                f"{type(self).__name__}.update takes the inputs {self.inputs}, got {len(args)} "
                f"positional arguments"
            )
        inputs = dict(zip(self.inputs, args))
        for name, value in kwargs.items():
            if name not in self.inputs:
                raise TypeError(
                    f"{type(self).__name__}.update takes the inputs {self.inputs}, got {name!r}"
                )
            if name in inputs:
                raise TypeError(f"{type(self).__name__}.update got multiple values for {name!r}")
            inputs[name] = value
        return inputs

    def _prepare(self, inputs: Dict[str, Tensor]) -> Optional[Dict[str, Tensor]]:
        """
        Checks the shapes, fixes the device, moves, flattens and casts the inputs, and removes
        void entries. Returns ``None`` for an empty batch.
        """
        tensors = {name: value for name, value in inputs.items() if value is not None}
        for name, value in tensors.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Input {name!r} must be a tensor, got {type(value).__name__}")
        shapes = {tuple(value.shape) for value in tensors.values()}
        if len(shapes) > 1:
            raise ValueError(
                "Inputs must have the same shape, got "
                + ", ".join(f"{name}: {tuple(value.shape)}" for name, value in tensors.items())
            )
        if next(iter(tensors.values())).numel() == 0:
            return None
        if self._device is None:
            self._device = next(iter(tensors.values())).device

        prepared = {}
        for name, value in tensors.items():
            dtype = torch.int32 if name in _LABEL_INPUTS else value.dtype
            flat = value.detach().reshape(-1)
            moved = flat.to(self._device, dtype)
            # stored inputs must not share memory with the caller's tensors, which may be
            # modified or reused after update
            prepared[name] = moved.clone() if moved.data_ptr() == flat.data_ptr() else moved

        if self.void_label is not None and "labels" in prepared:
            keep = prepared["labels"] != self.void_label
            prepared = {name: value[keep] for name, value in prepared.items()}
            if prepared["labels"].numel() == 0:
                return None
        return prepared


class StreamingMetric(Metric):
    """
    Metric that reduces each batch to a fixed-size state, e.g., counts, so its memory does not
    grow with the number of samples.
    """


class BufferedMetric(Metric):
    """
    Metric that needs all samples at once, e.g., the area under a curve. It stores the inputs
    until :meth:`~pytorch_ood.metrics.Metric.compute` is called. Labels and predictions are stored as int32.
    """

    def __init__(self, *, device: Optional[Device] = None, void_label: Optional[int] = None):
        super().__init__(device=device, void_label=void_label)
        self._buffers: Dict[str, List[Tensor]] = {name: [] for name in self.inputs}

    def _update(self, inputs: Dict[str, Tensor]) -> None:
        for name in self.inputs:
            self._buffers[name].append(inputs[name])

    def _compute(self) -> Dict[str, Tensor]:
        if not self._buffers[self.inputs[0]]:
            raise ValueError(f"{type(self).__name__} was given no data")
        return self._compute_from(_Data(_concat(self._buffers)))

    def _reset(self) -> None:
        self._buffers = {name: [] for name in self.inputs}

    @abstractmethod
    def _compute_from(self, data: _Data) -> Dict[str, Tensor]:
        """Computes the results from the complete inputs."""


def _concat(buffers: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
    for values in buffers.values():
        # keep the concatenation, so that later calls of compute do not copy the inputs again
        if len(values) > 1:
            values[:] = [torch.cat(values)]
    return {name: values[0] for name, values in buffers.items() if values}


class MetricCollection(Metric):
    """
    Computes several metrics from the same inputs, with the memory of a single metric: each
    input of the buffered metrics is stored once, streaming metrics only keep their state, and
    the curve that AUROC, AUPR and FPR@TPR are based on is computed once.

    :meth:`~pytorch_ood.metrics.Metric.update` takes the inputs as keyword arguments. Each metric receives the inputs it
    declares in :attr:`~pytorch_ood.metrics.Metric.inputs`.

    .. rubric:: Examples

    .. code-block:: python

        from pytorch_ood.metrics import AUROC, FPRAtTPR, Accuracy, MetricCollection

        metrics = MetricCollection([AUROC(), FPRAtTPR(0.9), Accuracy()])
        for x, y in loader:
            logits = model(x)
            metrics.update(scores=detector(x), labels=y, predictions=logits.argmax(dim=1))
        print(metrics.compute())  # {"AUROC": ..., "FPR90TPR": ..., "ACC": ...}
    """

    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        device: Optional[Device] = None,
        void_label: Optional[int] = None,
    ):
        """
        :param metrics: the metrics to compute. They must not set ``device`` or ``void_label``
            themselves, as the collection prepares the inputs for all of them.
        :param device: see :class:`~pytorch_ood.metrics.Metric`
        :param void_label: see :class:`~pytorch_ood.metrics.Metric`
        :raises ValueError: if ``metrics`` is empty, contains collections, metrics with their
            own ``device`` or ``void_label``, or metrics with the same keys
        """
        super().__init__(device=device, void_label=void_label)
        metrics = list(metrics)
        if not metrics:
            raise ValueError("A collection needs at least one metric")
        for metric in metrics:
            if isinstance(metric, MetricCollection):
                raise ValueError("Collections can not be nested")
            if metric.device is not None or metric.void_label is not None:
                raise ValueError(
                    f"{type(metric).__name__} sets its own device or void_label; set them on "
                    f"the collection instead"
                )
        keys = [key for metric in metrics for key in metric.keys]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"Several metrics compute the keys {duplicates}")

        self.metrics = metrics
        self._buffered = [m for m in metrics if isinstance(m, BufferedMetric)]
        self._streaming = [m for m in metrics if not isinstance(m, BufferedMetric)]
        self._buffer_inputs = tuple(
            dict.fromkeys(name for m in self._buffered for name in m.inputs)
        )
        self.inputs = tuple(dict.fromkeys(name for m in metrics for name in m.inputs))
        # metrics that are only computed if their inputs are given (see OODMetrics)
        self._optional: Tuple[Metric, ...] = ()
        self._fed: Dict[int, bool] = {}
        self._buffers: Dict[str, List[Tensor]] = {}
        self._reset()

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(key for metric in self.metrics for key in metric.keys)

    def update(self, *args: Tensor, **kwargs: Tensor) -> Self:
        """
        Adds a batch.

        :param kwargs: the inputs by name, see :attr:`~pytorch_ood.metrics.Metric.inputs` of the metrics
        :return: self
        :raises TypeError: if inputs are given as positional arguments
        :raises ValueError: if inputs that a metric needs are missing, or the inputs have
            different shapes
        """
        if args:
            raise TypeError("MetricCollection.update takes the inputs as keyword arguments")
        unknown = sorted(set(kwargs) - set(self.inputs))
        if unknown:
            raise TypeError(f"No metric of the collection takes the inputs {unknown}")
        given = {name for name, value in kwargs.items() if value is not None}

        fed_now = {}
        for metric in self.metrics:
            fed = all(name in given for name in metric.inputs)
            if metric in self._optional:
                previous = self._fed.get(id(metric))
                if previous is not None and previous != fed:
                    raise ValueError(
                        f"The inputs {metric.inputs} of {type(metric).__name__} must be given "
                        f"in all calls of update or in none"
                    )
                fed_now[id(metric)] = fed
            elif not fed:
                missing = [name for name in metric.inputs if name not in given]
                raise ValueError(f"{type(metric).__name__} needs the inputs {missing}")

        prepared = self._prepare(kwargs)
        # only after the inputs passed all checks
        self._fed.update(fed_now)
        if prepared is not None:
            self._update(prepared)
        return self

    def compute(self) -> Dict[str, float]:
        """
        Computes all metrics from the batches added so far.

        :return: the results of all metrics in one dictionary, in the order of ``metrics``,
            e.g., ``{"AUROC": ..., "FPR95TPR": ...}`` for ``[AUROC(), FPRAtTPR()]``
        :raises ValueError: if no data was given, or one of the metrics is undefined for the
            data
        """
        return super().compute()

    def _active(self, metrics: Sequence[Metric]) -> List[Metric]:
        return [m for m in metrics if m not in self._optional or self._fed.get(id(m), False)]

    def _update(self, inputs: Dict[str, Tensor]) -> None:
        for name in self._buffer_inputs:
            self._buffers[name].append(inputs[name])
        for metric in self._active(self._streaming):
            metric._update({name: inputs[name] for name in metric.inputs})

    def _compute(self) -> Dict[str, Tensor]:
        if self._buffer_inputs and not self._buffers[self._buffer_inputs[0]]:
            raise ValueError("The collection was given no data")
        data = _Data(_concat(self._buffers))
        results = {}
        for metric in self._active(self.metrics):
            if isinstance(metric, BufferedMetric):
                results.update(metric._compute_from(data))
            else:
                results.update(metric._compute())
        return results

    def _reset(self) -> None:
        self._buffers = {name: [] for name in self._buffer_inputs}
        self._fed = {}
        for metric in self.metrics:
            metric.reset()
