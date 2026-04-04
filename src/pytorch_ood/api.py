import logging
from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

Self = TypeVar("Self")
log = logging.getLogger(__name__)


class RequiresFittingException(Exception):
    """
    Raised when predict is called on a detector that has not been fitted.
    """

    def __init__(self, msg="You have to call fit() before predict()"):
        super(RequiresFittingException, self).__init__(msg)


class ModelNotSetException(ValueError):
    """
    Raised when predict() is called but no model was given.
    """

    def __init__(self, msg="When using predict(), model must not be None"):
        super(ModelNotSetException, self).__init__(msg)


class Detector(ABC):
    """
    Root public API for out-of-distribution detectors.

    Every detector supports ``predict(x)`` on raw model inputs and a generic
    ``fit(data_loader)`` entry point. Semantic subclasses refine
    this contract with alternate representation-specific methods such as
    ``predict_logits(...)`` or ``predict_features(...)``.
    """

    requires_fit = False  #: Whether ``fit(...)`` must be called before scoring.

    @staticmethod
    def _move_value_to_device(value, device: torch.device):
        """
        Move common detector-owned state to the given device.
        """
        if isinstance(value, Module):
            value.to(device)
            return value

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

    def to(self: Self, device) -> Self:
        """
        Move detector-owned modules and tensor state to ``device``.

        This is a detector-level analogue of ``nn.Module.to(...)``. It moves
        modules, tensors, and common container-valued state stored on the
        detector itself.

        :param device: target torch device
        :return: self
        """
        device = torch.device(device)
        self._device = device

        for attr, value in list(vars(self).items()):
            if attr == "_device":
                continue
            setattr(self, attr, self._move_value_to_device(value, device))

        return self

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Forwards to predict
        """
        return self.predict(*args, **kwargs)

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Fit the detector to a dataset. Some methods require this.

        :param data_loader: dataset to fit on. This is usually the training dataset.

        :raise ModelNotSetException: if model was not set
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

        :param x: batch of data
        :return: outlier scores for points

        :raise RequiresFitException: if detector has to be fitted to some data
        :raise ModelNotSetException: if model was not set
        """
        raise NotImplementedError


class LogitsDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes logits.

    Subclasses implement ``predict_logits(...)`` and optionally ``fit_logits(...)``.
    The default ``predict(x)`` and ``fit(data_loader)`` implementations
    forward raw inputs through ``self.model`` to obtain logits first.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        predict_logits = cls.__dict__.get("predict_logits")
        if predict_logits is not None:
            cls.predict_logits = cls._wrap_representation_method(predict_logits)

    @staticmethod
    def _wrap_representation_method(method):
        def wrapped(self, *args, **kwargs):
            device = self.device
            if device is not None:
                args, kwargs = self._move_tensor_arguments_to_device(args, kwargs, device)
            return method(self, *args, **kwargs)

        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        wrapped.__qualname__ = method.__qualname__
        return wrapped

    def predict(self, x: Tensor) -> Tensor:
        """
        Apply the model and forward its logits to ``predict_logits(...)``.

        :param x: input batch
        :return: outlier scores
        """
        if not hasattr(self, "model") or self.model is None:
            raise ModelNotSetException

        detector_device = self.device
        if detector_device is not None:
            x = x.to(detector_device)

        return self.predict_logits(self.model(x))

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Extract logits from a loader and forward them to ``fit_logits(...)``.

        :param data_loader: loader to extract logits from
        """
        if not hasattr(self, "model") or self.model is None:
            raise ModelNotSetException

        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        from .utils import extract_features

        z, y = extract_features(data_loader=data_loader, model=self.model, device=device)
        return self.fit_logits(z, y)

    def fit_logits(self: Self, logits: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on logits.

        :param logits: training logits to use for fitting.
        :param y: corresponding class labels.
        """
        raise NotImplementedError

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        Calculates outlier scores directly from logits.

        :param logits: batch of logits
        :return: outlier scores for points
        """
        raise NotImplementedError


class FeaturesDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes one feature tensor.

    Subclasses implement ``predict_features(...)`` and, when fitting is required,
    ``fit_features(...)``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        predict_features = cls.__dict__.get("predict_features")
        if predict_features is not None:
            cls.predict_features = cls._wrap_representation_method(predict_features)

    @staticmethod
    def _wrap_representation_method(method):
        def wrapped(self, *args, **kwargs):
            device = self.device
            if device is not None:
                args, kwargs = self._move_tensor_arguments_to_device(args, kwargs, device)
            return method(self, *args, **kwargs)

        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        wrapped.__qualname__ = method.__qualname__
        return wrapped

    def fit_features(self: Self, x: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on feature tensors.

        :param x: training features to use for fitting
        :param y: corresponding class labels
        """
        raise NotImplementedError

    def predict_features(self, x: Tensor) -> Tensor:
        """
        Calculate outlier scores directly from feature tensors.

        :param x: batch of features
        :return: outlier scores for points
        """
        raise NotImplementedError


class FeatureMapsDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes feature maps.

    Subclasses implement ``predict_feature_maps(...)`` and, when fitting is
    required, ``fit_feature_maps(...)``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        predict_feature_maps = cls.__dict__.get("predict_feature_maps")
        if predict_feature_maps is not None:
            cls.predict_feature_maps = cls._wrap_representation_method(predict_feature_maps)

    @staticmethod
    def _wrap_representation_method(method):
        def wrapped(self, *args, **kwargs):
            device = self.device
            if device is not None:
                args, kwargs = self._move_tensor_arguments_to_device(args, kwargs, device)
            return method(self, *args, **kwargs)

        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        wrapped.__qualname__ = method.__qualname__
        return wrapped

    def fit_feature_maps(self: Self, feature_maps: Tensor, y: Tensor) -> Self:
        """
        Fit the detector directly on feature maps.

        :param feature_maps: training feature maps to use for fitting.
        :param y: corresponding class labels.
        """
        raise NotImplementedError

    def predict_feature_maps(self, feature_maps: Tensor) -> Tensor:
        """
        Calculates outlier scores directly from feature maps.

        :param feature_maps: batch of feature maps
        :return: outlier scores for points
        """
        raise NotImplementedError


class StructuredDetector(Detector):
    """
    Base class for detectors whose alternate public API consumes structured inputs.

    This is used for detectors whose non-model interface is not well described
    by a single tensor family, for example lists of per-layer features or mixed
    inputs such as logits plus feature maps.
    """

    def fit_structured(self: Self, *args, **kwargs) -> Self:
        """
        Fit the detector directly on structured intermediate representations.
        """
        raise NotImplementedError

    def predict_structured(self, *args, **kwargs) -> Tensor:
        """
        Calculates outlier scores directly from structured intermediate representations.
        """
        raise NotImplementedError
