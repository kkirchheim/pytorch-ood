import functools
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ClassVar, Dict, FrozenSet, Iterable, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from typing_extensions import Self

log = logging.getLogger(__name__)

_NAMED = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


class RequiresFittingException(Exception):
    """
    Raised when predict is called on a detector that has not been fitted.
    """

    def __init__(self, msg="You have to call fit() before predict()"):
        super(RequiresFittingException, self).__init__(msg)


class ModelNotSetException(ValueError):
    """
    Raised when predict() or fit() needs a model but none was given.
    """

    def __init__(self, msg="When using predict(), model must not be None"):
        super(ModelNotSetException, self).__init__(msg)


# -- Component metadata --------------------------------------------------------
# Detectors, training objectives, datasets and benchmarks carry a class attribute
# ``info`` with one of the records below. The documentation renders badges, links and
# comparison tables from it, and the tests check the claims it makes.


class Task(str, Enum):
    """
    Granularity at which inputs are labeled or scored as in- or out-of-distribution.
    """

    #: one label or outlier score per input, e.g. a :math:`B \times C` logit tensor
    CLASSIFICATION = "classification"
    #: one label or outlier score per pixel, e.g. a :math:`B \times C \times H \times W` logit tensor
    SEGMENTATION = "segmentation"


class Representation(str, Enum):
    """
    What a component is applied to. For detectors, this follows from their base
    class (see :doc:`/core_api/detectors`); training objectives list it in their ``info``.
    """

    #: model outputs before the softmax
    LOGITS = "logits"
    #: pooled features of the penultimate layer, shape :math:`B \times D`
    FEATURES = "features"
    #: activations of a convolutional layer, shape :math:`B \times C \times H \times W`
    FEATURE_MAPS = "feature maps"
    #: activations of several layers at once
    LAYERS = "several layers"
    #: distances to class centers
    DISTANCES = "distances"
    #: a separately predicted confidence
    CONFIDENCE = "confidence"
    #: the model's original inputs, e.g. to compute gradients with respect to them or to
    #: sample with dropout
    INPUTS = "model inputs"


class Role(str, Enum):
    """
    The part a dataset usually plays in OOD detection experiments.
    """

    #: in-distribution data that models are trained and evaluated on
    IN_DISTRIBUTION = "in-distribution"
    #: out-of-distribution data used only for evaluation
    OOD_TEST = "OOD test set"
    #: in-distribution data under distribution shift, e.g. corruptions
    DISTRIBUTION_SHIFT = "distribution shift"
    #: example outliers used during training, e.g. for Outlier Exposure
    AUXILIARY_OUTLIERS = "auxiliary outliers"
    #: in- and out-of-distribution samples in one dataset, e.g. anomaly segmentation benchmarks
    BENCHMARK = "benchmark"


@dataclass(frozen=True)
class Paper:
    """
    The publication that introduced a component.
    """

    #: title of the publication
    title: str
    #: short venue name, e.g. ``"NeurIPS"`` or ``"arXiv"``
    venue: str
    #: year of publication
    year: int
    #: link to the publication
    url: str
    #: reference implementation by the authors, if public
    code: Optional[str] = None


def _frozen(values: Iterable) -> FrozenSet:
    return values if isinstance(values, frozenset) else frozenset(values)


@dataclass(frozen=True)
class DetectorInfo:
    """
    Metadata of an OOD detector, see :attr:`pytorch_ood.api.Detector.info`.
    """

    #: tasks the detector supports; checked by the test suite
    tasks: FrozenSet[Task]
    #: publication that introduced the detector; ``None`` for baselines without one
    paper: Optional[Paper] = None
    #: whether the implementation was written with an AI coding assistant
    ai_coded: bool = False

    def __post_init__(self):
        object.__setattr__(self, "tasks", _frozen(self.tasks))


@dataclass(frozen=True)
class LossInfo:
    """
    Metadata of a training objective.
    """

    #: tasks the objective supports; checked by the test suite
    tasks: FrozenSet[Task]
    #: what the objective is applied to, e.g. logits, or logits and features
    inputs: FrozenSet[Representation]
    #: whether the objective takes example outliers (targets :math:`< 0`, e.g. auxiliary OOD
    #: data) during training; unsupervised objectives only take in-distribution data, possibly
    #: synthesizing outliers from it internally
    supervised: bool
    #: publication that introduced the objective; ``None`` for baselines without one
    paper: Optional[Paper] = None
    #: whether the implementation was written with an AI coding assistant
    ai_coded: bool = False

    def __post_init__(self):
        object.__setattr__(self, "tasks", _frozen(self.tasks))
        object.__setattr__(self, "inputs", _frozen(self.inputs))


@dataclass(frozen=True)
class DatasetInfo:
    """
    Metadata of a dataset.
    """

    #: granularity of the labels
    task: Task
    #: SPDX identifier (e.g. ``"CC-BY-4.0"``) or short description of the license or terms
    #: of use; ``None`` if no license statement is known
    license: Optional[str]
    #: parts the dataset usually plays in OOD experiments; empty where no usage is established
    roles: FrozenSet[Role] = frozenset()
    #: publication that introduced the dataset, if any
    paper: Optional[Paper] = None
    #: project page of the dataset, if any
    homepage: Optional[str] = None
    #: whether the implementation was written with an AI coding assistant
    ai_coded: bool = False

    def __post_init__(self):
        object.__setattr__(self, "roles", _frozen(self.roles))


@dataclass(frozen=True)
class BenchmarkInfo:
    """
    Metadata of a benchmark, see :attr:`pytorch_ood.benchmark.Benchmark.info`.
    """

    #: publication whose evaluation protocol the benchmark reproduces
    paper: Paper
    #: tasks the benchmark evaluates
    tasks: FrozenSet[Task]
    #: whether the implementation was written with an AI coding assistant
    ai_coded: bool = False

    def __post_init__(self):
        object.__setattr__(self, "tasks", _frozen(self.tasks))


# -- Detectors ------------------------------------------------------------------


def _wrap_predict_method(cls: type, name: str) -> None:
    """
    Wraps the method ``name`` if ``cls`` itself defines it, so that it runs under
    :class:`torch.no_grad`, which makes scores non-differentiable, and receives its tensor
    arguments on the detector's device, if the detector has one. Tensors nested in lists are not
    moved.
    """
    method = cls.__dict__.get(name)
    if method is None:
        return

    # functools.wraps also sets __wrapped__, so that inspect.signature (and the docs) show
    # the parameters of the wrapped method instead of *args, **kwargs
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        device = self.device
        if device is not None:
            args, kwargs = self._move_tensor_arguments_to_device(args, kwargs, device)
        with torch.no_grad():
            return method(self, *args, **kwargs)

    setattr(cls, name, wrapped)


def _wrap_fit_method(cls: type, name: str) -> None:
    """
    Wraps the method ``name(representation, labels, ...)`` if ``cls`` itself defines it, so that
    it receives only the ID rows of both, unless the detector sets ``_fit_uses_ood``. Warns if
    rows were dropped and raises :class:`ValueError` if no ID row is left. Gradients stay enabled
    and arguments stay on their device.

    :raises TypeError: if the method does not name these two parameters
    """
    method = cls.__dict__.get(name)
    if method is None:
        return

    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())[1:3]
    if len(parameters) < 2 or any(p.kind not in _NAMED for p in parameters):
        raise TypeError(
            f"{method.__qualname__} must take the representation and the labels as its first "
            f"two parameters, e.g. fit_features(self, z, y), got {signature}"
        )
    representation, labels = (p.name for p in parameters)

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        y = bound.arguments.get(labels)
        if not self._fit_uses_ood and isinstance(y, Tensor):
            from .utils.utils import _id_rows_for_fit

            bound.arguments[representation], bound.arguments[labels] = _id_rows_for_fit(
                bound.arguments[representation], y
            )
        return method(*bound.args, **bound.kwargs)

    setattr(cls, name, wrapped)


class Detector(ABC):
    """
    Root public API for out-of-distribution detectors.

    Every detector supports ``predict(x)`` on raw model inputs and a generic
    ``fit(data_loader)`` entry point. Semantic subclasses refine
    this contract with alternate representation-specific methods such as
    ``predict_logits(...)`` or ``predict_features(...)``.
    """

    requires_fit = False  #: Whether ``fit(...)`` must be called before scoring.

    hyperparameter_space: Dict[str, List] = {}
    """
    Search space for hyperparameter optimization, mapping each tunable hyperparameter
    name to the list of candidate values to try. Empty for detectors without tunable
    hyperparameters. Used by :class:`pytorch_ood.utils.GridSearch`.
    """

    info: ClassVar[Optional[DetectorInfo]] = None
    """
    Metadata of the detector: the paper that introduced it, the tasks it supports,
    and so on (see :doc:`/core_api/metadata`). Every detector in
    :mod:`pytorch_ood.detector` defines its own.
    """

    #: The attribute that computes the representation from raw inputs (``model``, ``encoder``,
    #: ``backbone``), or ``None`` if there is no single one.
    _producer_name: ClassVar[Optional[str]] = None

    #: Whether fitting uses OOD samples (labels below zero). If ``False``, ``fit(...)`` and
    #: ``fit_<kind>(...)`` ignore them with a warning, and raise :class:`ValueError` if no ID
    #: sample is left.
    _fit_uses_ood: ClassVar[bool] = False

    def _require_producer(self) -> Callable[[Tensor], Tensor]:
        """
        The producer (see ``_producer_name``).

        :raises ModelNotSetException: if it is ``None``
        """
        producer = getattr(self, self._producer_name, None)
        if producer is None:
            raise ModelNotSetException
        return producer

    def _fit_device(self) -> Union[str, torch.device]:
        """
        The detector's device, or ``"cpu"`` (with a warning, and the detector moved there) if it
        has none.
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)
        return device

    def _predict_produced(self, x: Tensor, predict: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Moves ``x`` to the detector's device, computes its representation with the producer
        under :func:`torch.no_grad` and scores it with ``predict``.
        """
        producer = self._require_producer()
        device = self.device
        if device is not None:
            x = x.to(device)
        with torch.no_grad():
            z = producer(x)
        return predict(z)

    def _fit_produced(
        self, data_loader: DataLoader, fit: Callable[[Tensor, Tensor], Self]
    ) -> Self:
        """
        Extracts the representations of the ID samples in ``data_loader`` with the producer and
        calls ``fit`` on them. Does nothing if the detector requires no fitting.
        """
        if not self.requires_fit:
            return self
        producer = self._require_producer()

        from .utils import extract_features

        z, y = extract_features(data_loader, producer, device=self._fit_device())
        return fit(z, y)

    @staticmethod
    def _move_value_to_device(value, device: torch.device):
        """
        Move common detector-owned state to the given device.
        """
        if isinstance(value, Module):
            value.to(device)
            return value

        if isinstance(value, Parameter):
            return Parameter(value.detach().to(device), requires_grad=value.requires_grad)

        if isinstance(value, Tensor):
            return value.to(device)

        owner = getattr(value, "__self__", None)
        if isinstance(owner, Module):
            owner.to(device)
            return value

        if isinstance(value, list):
            return [Detector._move_value_to_device(v, device) for v in value]

        if isinstance(value, tuple):
            return tuple(Detector._move_value_to_device(v, device) for v in value)

        if isinstance(value, dict):
            for key, inner in value.items():
                value[key] = Detector._move_value_to_device(inner, device)
            return value

        return value

    @staticmethod
    def _move_tensor_arguments_to_device(args, kwargs, device: torch.device):
        """
        Move tensor-valued positional and keyword arguments to ``device``.
        """
        moved_args = tuple(
            value.to(device) if isinstance(value, Tensor) else value for value in args
        )
        moved_kwargs = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in kwargs.items()
        }
        return moved_args, moved_kwargs

    @staticmethod
    def _infer_value_device(value):
        """
        Infer the device of common detector-owned state, if any.
        """
        if isinstance(value, Module):
            try:
                return next(value.parameters()).device
            except StopIteration:
                try:
                    return next(value.buffers()).device
                except StopIteration:
                    return None

        if isinstance(value, Tensor):
            return value.device

        owner = getattr(value, "__self__", None)
        if isinstance(owner, Module):
            return Detector._infer_value_device(owner)

        if isinstance(value, (list, tuple)):
            for inner in value:
                device = Detector._infer_value_device(inner)
                if device is not None:
                    return device
            return None

        if isinstance(value, dict):
            for inner in value.values():
                device = Detector._infer_value_device(inner)
                if device is not None:
                    return device
            return None

        return None

    @property
    def device(self):
        """
        The device of the detector's owned torch state, if one can be inferred.
        """
        for value in vars(self).values():
            device = self._infer_value_device(value)
            if device is not None:
                return device

        return getattr(self, "_device", None)

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move detector-owned modules and tensor state to ``device``.

        This is a detector-level analogue of ``nn.Module.to(...)``. It moves
        modules, tensors, and common container-valued state stored on the
        detector itself.

        :param device: target device, e.g. ``"cuda:0"`` or a :class:`torch.device`
        :return: self
        """
        device = torch.device(device)
        self._device = device

        for attr, value in list(vars(self).items()):
            if attr == "_device":
                continue
            setattr(self, attr, self._move_value_to_device(value, device))

        return self

    def get_hyperparameters(self) -> Dict:
        """
        Return the detector's current tunable hyperparameter values.

        The default implementation reads, for each key in
        :attr:`hyperparameter_space`, the attribute of the same name. Detectors
        whose hyperparameters are not stored as plain attributes of that name
        should override this together with :meth:`set_hyperparameters`.

        :return: mapping from hyperparameter name to current value
        """
        return {name: getattr(self, name) for name in self.hyperparameter_space}

    def set_hyperparameters(self, **kwargs) -> Self:
        """
        Set tunable hyperparameters by name.

        The default implementation assigns each value to the attribute of the
        same name. Detectors whose hyperparameters require derived state (for
        example a percentile that must be turned into an activation threshold)
        should override this.

        :param kwargs: hyperparameter values to set; keys must be in
            :attr:`hyperparameter_space`
        :return: self
        :raises ValueError: if a key is not a known hyperparameter
        """
        for name, value in kwargs.items():
            if name not in self.hyperparameter_space:
                raise ValueError(
                    f"Unknown hyperparameter '{name}' for {type(self).__name__}. "
                    f"Known hyperparameters: {list(self.hyperparameter_space)}"
                )
            setattr(self, name, value)

        return self

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Fit the detector to a dataset. Some methods require this.

        The base implementation does nothing and returns ``self`` for detectors
        with ``requires_fit = False``. Detectors that require fitting override it.

        :param data_loader: data loader to fit on. This is usually the training set,
            yielding batches ``(x, y)``.
        :return: the fitted detector
        :raises NotImplementedError: if the detector requires fitting but does not override ``fit()``
        """
        if self.requires_fit:
            raise NotImplementedError(
                f"{type(self).__name__} requires fitting and must implement fit()."
            )

        return self

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """
        Calculates outlier scores. Inputs will be passed through the model.

        :param x: batch of inputs, e.g. images of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B` (:math:`B \\times H \\times W` for grid-like
            input, e.g. in anomaly segmentation). Larger values indicate that the input is more
            likely to be out-of-distribution.

        :raises RequiresFittingException: if detector has to be fitted to some data
        :raises ModelNotSetException: if model was not set
        """
        raise NotImplementedError


class LogitsDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes logits.

    Subclasses implement ``predict_logits(...)`` and optionally ``fit_logits(...)``.
    The default ``predict(x)`` and ``fit(data_loader)`` implementations
    forward raw inputs through ``self.model`` to obtain logits first.
    """

    _producer_name = "model"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_predict_method(cls, "predict_logits")
        _wrap_fit_method(cls, "fit_logits")

    def predict(self, x: Tensor) -> Tensor:
        """
        Apply the model and forward its logits to ``predict_logits(...)``.

        ``x`` is moved to the detector's device (if one is set). The model is called as
        is; it is not switched to evaluation mode.

        :param x: input batch, e.g. images of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B` (:math:`B \\times H \\times W` for grid-like
            input)
        :raises ModelNotSetException: if the detector has no model
        """
        return self._predict_produced(x, self.predict_logits)

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extract logits of the ID samples in a loader with the detector's model and forward them
        to ``fit_logits(...)``. Does nothing if the detector requires no fitting.

        :param data_loader: loader yielding batches ``(x, y)``; OOD samples are ignored
        :return: the fitted detector
        :raises ModelNotSetException: if the detector has no model
        :raises ValueError: if the loader contains no ID samples
        """
        return self._fit_produced(data_loader, self.fit_logits)

    def fit_logits(self, logits: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on logits.

        :param logits: training logits of shape :math:`N \\times C`
        :param y: corresponding class labels of shape :math:`N`
        :return: the fitted detector
        """
        if not self.requires_fit:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} requires fitting but fit_logits() is not implemented"
        )

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        Calculates outlier scores directly from logits.

        :param logits: logits of shape :math:`B \\times C` (or :math:`B \\times C \\times H \\times W`
            for grid-like input)
        :return: outlier scores of shape :math:`B` (:math:`B \\times H \\times W` for grid-like
            input), larger values indicate that the input is more likely to be out-of-distribution
        """
        raise NotImplementedError


class FeaturesDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes one feature tensor.

    Subclasses implement ``predict_features(...)`` and, when fitting is required,
    ``fit_features(...)``. The default ``predict(x)`` and ``fit(data_loader)`` implementations
    forward raw inputs through ``self.encoder`` to obtain features first.

    **Parameter naming convention**: Subclasses that accept a feature extractor should use
    the parameter name ``encoder`` to receive a callable that produces pooled feature vectors
    of shape :math:`B \\times D`, where :math:`B` is batch size and :math:`D` is feature dimension.
    """

    _producer_name = "encoder"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_predict_method(cls, "predict_features")
        _wrap_fit_method(cls, "fit_features")

    def predict(self, x: Tensor) -> Tensor:
        """
        Apply the encoder and forward its features to ``predict_features(...)``.

        ``x`` is moved to the detector's device (if one is set). The encoder is called as
        is; it is not switched to evaluation mode.

        :param x: input batch, e.g. images of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if the detector has no encoder
        :raises RequiresFittingException: if the detector requires fitting and was not fitted
        """
        return self._predict_produced(x, self.predict_features)

    def fit(self, data_loader: DataLoader) -> Self:
        """
        Extract features of the ID samples in a loader with the detector's encoder and forward
        them to ``fit_features(...)``. Does nothing if the detector requires no fitting.

        :param data_loader: loader yielding batches ``(x, y)``; OOD samples are ignored
        :return: the fitted detector
        :raises ModelNotSetException: if the detector has no encoder
        :raises ValueError: if the loader contains no ID samples
        """
        return self._fit_produced(data_loader, self.fit_features)

    def fit_features(self, x: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on feature tensors.

        :param x: training features of shape :math:`N \\times D`
        :param y: corresponding class labels of shape :math:`N`
        :return: the fitted detector
        """
        if not self.requires_fit:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} requires fitting but fit_features() is not implemented"
        )

    def predict_features(self, x: Tensor) -> Tensor:
        """
        Calculate outlier scores directly from feature tensors.

        :param x: pooled features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`, larger values indicate that the input is more
            likely to be out-of-distribution
        """
        raise NotImplementedError


class FeatureMapsDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes feature maps.

    Subclasses implement ``predict_feature_maps(...)`` and, when fitting is
    required, ``fit_feature_maps(...)`` and ``fit(data_loader)``. The default ``predict(x)``
    forwards raw inputs through ``self.backbone`` to obtain feature maps first; there is no
    default ``fit(data_loader)``.

    **Parameter naming convention**: Subclasses that accept a feature extractor should use
    the parameter name ``backbone`` to receive a callable that produces spatial feature maps
    of shape :math:`B \\times C \\times H \\times W`, where :math:`B` is batch size, :math:`C` is number of
    channels, and :math:`H, W` are spatial dimensions.
    """

    _producer_name = "backbone"
    # no default fit: extract_features flattens each sample, which would lose the C x H x W shape
    # that fit_feature_maps needs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_predict_method(cls, "predict_feature_maps")
        _wrap_fit_method(cls, "fit_feature_maps")

    def predict(self, x: Tensor) -> Tensor:
        """
        Apply the backbone and forward its feature maps to ``predict_feature_maps(...)``.

        ``x`` is moved to the detector's device (if one is set). The backbone is called as
        is; it is not switched to evaluation mode.

        :param x: input batch, e.g. images of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B`
        :raises ModelNotSetException: if the detector has no backbone
        :raises RequiresFittingException: if the detector requires fitting and was not fitted
        """
        return self._predict_produced(x, self.predict_feature_maps)

    def fit_feature_maps(self, feature_maps: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on feature maps.

        :param feature_maps: training feature maps of shape :math:`N \\times C \\times H \\times W`
        :param y: corresponding class labels of shape :math:`N`
        :return: the fitted detector
        """
        if not self.requires_fit:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} requires fitting but fit_feature_maps() is not implemented"
        )

    def predict_feature_maps(self, feature_maps: Tensor) -> Tensor:
        """
        Calculates outlier scores directly from feature maps.

        :param feature_maps: feature maps of shape :math:`B \\times C \\times H \\times W`
        :return: outlier scores of shape :math:`B`, larger values indicate that the input is more
            likely to be out-of-distribution
        """
        raise NotImplementedError


class StructuredDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes structured inputs.

    This is used for detectors whose non-model interface is not well described
    by a single tensor family, for example lists of per-layer features or mixed
    inputs such as logits plus feature maps.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _wrap_predict_method(cls, "predict_structured")
        # fit_structured has no fixed signature: only the detector knows which argument holds
        # the labels, so it drops the OOD rows itself (see _id_rows_for_fit)

    def fit_structured(self, *args, **kwargs) -> Self:
        """
        Fit the detector directly on structured intermediate representations.

        The expected arguments depend on the detector, see its documentation.

        :return: the fitted detector
        """
        if not self.requires_fit:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} requires fitting but fit_structured() is not implemented"
        )

    def predict_structured(self, *args, **kwargs) -> Tensor:
        """
        Calculates outlier scores directly from structured intermediate representations.

        The expected arguments depend on the detector, see its documentation.

        :return: outlier scores with one entry per input, larger values indicate that the input is
            more likely to be out-of-distribution
        """
        raise NotImplementedError


class GradientDetector(Detector):
    """
    Base class for detectors that require gradient computation during prediction.

    Unlike feature- or logit-based detectors that operate under ``torch.no_grad()``,
    these detectors compute gradients (input-space, parameter-space, or
    activation-space) as part of the scoring process. Callers must ensure that
    the model and inputs are in a state that allows gradient computation.
    """

    pass
