import hashlib
import inspect
import os
import pickle
import re
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Sized, Tuple, Union, overload

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_ood.api import (
    BenchmarkInfo,
    Detector,
    FeaturesDetector,
    GradientDetector,
    LogitsDetector,
)
from pytorch_ood.metrics import OODMetrics
from pytorch_ood.utils import TensorBuffer

_CACHE_VERSION = 2

# torch.load gained weights_only in 1.13. A cache directory can be shared, and pickle runs code.
_SAFE_LOAD = (
    {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
)


def _sanitize_cache_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "cache"


class Benchmark(ABC):
    """
    Base class for benchmarks.

    A benchmark provides an in-distribution training set via :meth:`train_set` and a list of
    test sets via :meth:`test_sets`, one per OOD dataset. In the test sets, ID samples carry
    their class labels and OOD samples are labelled with values :math:`< 0`. Subclasses set the
    attribute ``ood_names``, a list with the name of each OOD dataset in the order of
    :meth:`test_sets`. These names are the values of the ``Dataset`` field in the results of
    :meth:`evaluate`.
    """

    info: ClassVar[Optional[BenchmarkInfo]] = None
    """
    Metadata of the benchmark: the paper whose evaluation protocol it reproduces,
    and so on (see :doc:`/core_api/metadata`).
    """

    @abstractmethod
    def train_set(self) -> Dataset:
        """
        In-distribution training dataset, e.g. to fit a detector or train a model.
        """

    @abstractmethod
    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.

        :param known: include ID
        :param unknown: include OOD
        :return: with ``known`` and ``unknown``, one dataset per entry of ``ood_names`` that
            combines the ID test set with that OOD set. With ``unknown`` only, the OOD sets in the
            order of ``ood_names``. With ``known`` only, the ID test set, once. OOD samples are
            labelled :math:`< 0`.
        :raises ValueError: if both ``known`` and ``unknown`` are false
        """
        pass

    def _ensure_cache_state(self):
        if not hasattr(self, "_representation_cache"):
            self._representation_cache = {}

        if not hasattr(self, "_cache_warnings_shown"):
            self._cache_warnings_shown = set()

    def _warn_once(self, key: str, message: str, stacklevel: int = 3) -> None:
        self._ensure_cache_state()
        if key in self._cache_warnings_shown:
            return

        warnings.warn(message, UserWarning, stacklevel=stacklevel)
        self._cache_warnings_shown.add(key)

    @staticmethod
    def _normalize_detectors(
        detector: Union[Detector, Sequence[Detector]],
    ) -> Tuple[List[Detector], bool]:
        if isinstance(detector, Detector):
            return [detector], False

        if isinstance(detector, Sequence):
            detectors = list(detector)
            if not detectors:
                raise ValueError("At least one detector must be provided")
            if not all(isinstance(item, Detector) for item in detectors):
                raise TypeError("All elements must be Detector instances")
            return detectors, True

        raise TypeError("detector must be a Detector or a sequence of Detector instances")

    @staticmethod
    def _get_logits_producer(detector: LogitsDetector):
        return getattr(detector, "model", None)

    @staticmethod
    def _get_features_producer(detector: FeaturesDetector):
        return getattr(detector, "encoder", None)

    @staticmethod
    def _producer_identity(producer, known: Dict) -> Optional[str]:
        """
        Digest of what determines the outputs of ``producer``: the class and method of its module,
        and the train/eval mode and tensors of the module and its submodules.

        :param producer: a module or a bound method of one
        :param known: digests computed so far in this call of ``evaluate``
        :return: the digest, ``None`` if ``producer`` has no module or the module has no tensors
        """
        module = getattr(producer, "__self__", producer)
        if not isinstance(module, torch.nn.Module):
            return None

        func = getattr(producer, "__func__", None)
        method = "__call__" if func is None else func.__qualname__
        call = (id(module), method)
        if call not in known:
            tensors = list(chain(module.named_parameters(), module.named_buffers()))
            if not tensors:
                known[call] = None
                return None

            cls = type(module)
            modes = [m.training for m in module.modules()]
            digest = hashlib.sha256(
                f"{cls.__module__}.{cls.__qualname__}.{method}.{modes}".encode()
            )
            for name, tensor in tensors:
                data = tensor.detach().cpu()
                if data.dtype == torch.bfloat16:  # numpy has no such type
                    data = data.float()
                digest.update(f"{name}.{data.dtype}.{tuple(data.shape)}".encode())
                digest.update(data.numpy().tobytes())
            known[call] = digest.hexdigest()[:16]
        return known[call]

    @staticmethod
    def _cache_file_path(cache_dir: str, entry: Tuple) -> Path:
        return Path(cache_dir) / ("_".join(_sanitize_cache_token(str(p)) for p in entry) + ".pt")

    def _load_disk_cache(
        self, path: Path, entry: Tuple, num_samples: Optional[int]
    ) -> Optional[Dict]:
        if not path.exists():
            return None

        try:
            payload = torch.load(path, map_location="cpu", **_SAFE_LOAD)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            # called from _get_representation_cache, 6 frames below the caller of evaluate
            self._warn_once(f"unreadable:{path}", f"Ignoring unreadable cache {path}: {e}", 6)
            return None

        metadata = payload.get("metadata", {})
        if metadata.get("cache_version") != _CACHE_VERSION or metadata.get("entry") != list(entry):
            return None
        if num_samples is not None and metadata.get("num_samples") != num_samples:
            return None

        return payload

    @staticmethod
    def _extract_representation(
        data_loader: DataLoader,
        producer,
        device: str,
        representation: str,
    ) -> Dict:
        buffer = TensorBuffer(device="cpu")

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                z = producer(x)
                z = z.view(z.shape[0], -1)
                buffer.append(representation, z)
                buffer.append("label", y)

        return {
            representation: buffer.get(representation),
            "label": buffer.get("label"),
        }

    def _get_representation_cache(
        self,
        split: str,
        dataset_name: str,
        representation: str,
        producer,
        data_loader: DataLoader,
        device: str,
        cache: bool,
        cache_dir: Optional[str],
        cache_key: Optional[str],
        local_cache: Dict,
        identities: Dict,
    ) -> Dict:
        self._ensure_cache_state()
        # a bound method is a new object on every access, so its id says nothing: use its parts
        owner = getattr(producer, "__self__", producer)
        local_key = (representation, id(owner), getattr(producer, "__func__", None))
        if local_key in local_cache:
            return local_cache[local_key]

        identity = None
        if cache or cache_dir is not None:
            identity = self._producer_identity(producer, identities)
            if identity is None:
                self._warn_once(
                    "unidentified_producer",
                    "The outputs of a model or encoder that has no parameters or buffers, or is "
                    "not a module or a method of one, cannot be identified and are not cached.",
                    stacklevel=5,
                )

        entry = None
        if identity is not None:
            entry = (
                type(self).__qualname__,
                split,
                dataset_name,
                representation,
                identity,
                cache_key,
            )

        if cache and entry in self._representation_cache:
            local_cache[local_key] = self._representation_cache[entry]
            return local_cache[local_key]

        path = (
            None if entry is None or cache_dir is None else self._cache_file_path(cache_dir, entry)
        )
        dataset = data_loader.dataset
        num_samples = len(dataset) if isinstance(dataset, Sized) else None
        payload = None if path is None else self._load_disk_cache(path, entry, num_samples)

        if payload is None:
            data = self._extract_representation(
                data_loader=data_loader,
                producer=producer,
                device=device,
                representation=representation,
            )
            payload = {"data": data}
            if path is not None:
                payload["metadata"] = {
                    "cache_version": _CACHE_VERSION,
                    "entry": list(entry),
                    "num_samples": int(data["label"].shape[0]),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                # an interrupted write must not leave a truncated entry behind
                part = path.with_name(path.name + ".part")
                torch.save(payload, part)
                os.replace(part, path)

        if cache and entry is not None:
            self._representation_cache[entry] = payload
        local_cache[local_key] = payload
        return payload

    @staticmethod
    def _supports_cached_logits(detector: Detector) -> bool:
        return (
            isinstance(detector, LogitsDetector) and getattr(detector, "model", None) is not None
        )

    @staticmethod
    def _supports_cached_features(detector: Detector) -> bool:
        if not isinstance(detector, FeaturesDetector):
            return False

        if isinstance(detector, GradientDetector):
            return False

        return True

    @staticmethod
    def _evaluate_raw(detector: Detector, data_loader: DataLoader, device: str) -> Dict:
        metrics = OODMetrics()

        for x, y in data_loader:
            y = y.to(device)
            scores = detector(x.to(device))
            metrics.update(scores, y.to(scores.device))

        return metrics.compute()

    @staticmethod
    def _evaluate_logits(detector: LogitsDetector, payload: Dict) -> Dict:
        metrics = OODMetrics()
        logits = payload["data"]["logits"]
        labels = payload["data"]["label"]
        scores = detector.predict_logits(logits)
        predictions = logits.argmax(dim=1).to(scores.device)
        metrics.update(scores, labels.to(scores.device), predictions)
        return metrics.compute()

    @staticmethod
    def _evaluate_features(detector: FeaturesDetector, payload: Dict) -> Dict:
        metrics = OODMetrics()
        features = payload["data"]["features"]
        labels = payload["data"]["label"]
        scores = detector.predict_features(features)
        metrics.update(scores, labels.to(scores.device))
        return metrics.compute()

    def _evaluate_single_detector(
        self,
        detector: Detector,
        dataset_name: str,
        data_loader: DataLoader,
        device: str,
        cache: bool,
        cache_dir: Optional[str],
        cache_key: Optional[str],
        local_cache: Dict,
        identities: Dict,
    ) -> Dict:
        detector = detector.to(device)

        if self._supports_cached_logits(detector):
            producer = self._get_logits_producer(detector)
            payload = self._get_representation_cache(
                split="eval",
                dataset_name=dataset_name,
                representation="logits",
                producer=producer,
                data_loader=data_loader,
                device=device,
                cache=cache,
                cache_dir=cache_dir,
                cache_key=cache_key,
                local_cache=local_cache,
                identities=identities,
            )
            return self._evaluate_logits(detector, payload)

        if self._supports_cached_features(detector):
            producer = self._get_features_producer(detector)
            if producer is not None:
                payload = self._get_representation_cache(
                    split="eval",
                    dataset_name=dataset_name,
                    representation="features",
                    producer=producer,
                    data_loader=data_loader,
                    device=device,
                    cache=cache,
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    local_cache=local_cache,
                    identities=identities,
                )
                return self._evaluate_features(detector, payload)

        return self._evaluate_raw(detector, data_loader, device)

    @overload
    def evaluate(
        self,
        detector: Detector,
        loader_kwargs: Optional[Dict] = None,
        device: str = "cpu",
        cache: bool = False,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> List[Dict]: ...

    @overload
    def evaluate(
        self,
        detector: Sequence[Detector],
        loader_kwargs: Optional[Dict] = None,
        device: str = "cpu",
        cache: bool = False,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> List[Dict]: ...

    def evaluate(
        self,
        detector: Union[Detector, Sequence[Detector]],
        loader_kwargs: Optional[Dict] = None,
        device: str = "cpu",
        cache: bool = False,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Evaluate one detector or a list of detectors on all benchmark datasets.

        When several logits detectors or pooled-feature detectors are evaluated
        together, the outputs of a model are computed once and shared. With ``cache=True``,
        they are also kept on the benchmark instance for later ``evaluate(...)`` calls. With
        ``cache_dir``, they are stored on disk. The two options are independent.

        Cached outputs are identified by the benchmark, the OOD dataset, the class of the model,
        the parameters, buffers and train/eval mode of the model and its submodules, the method
        that produces them, and ``cache_key``. Other attributes of the model, the transforms and
        the data are not detected, so change ``cache_key`` when they change. A ``model`` or
        ``encoder`` that has no parameters or buffers, or is neither a module nor a method of
        one, such as a lambda, cannot be identified. Its outputs are only shared within one call,
        and a warning is issued if caching was requested.

        Only logits detectors (using ``model``) and pooled-feature detectors (using
        ``encoder``) can reuse cached representations; all other detectors run the full
        pipeline. Detectors are not fitted by this method, call ``fit()`` beforehand. They are
        moved to ``device``. This method does not change the train or eval mode of their
        ``model``, ``encoder`` or ``backbone``, and warns once per benchmark if one of them is in
        training mode. The metrics are those computed by
        :class:`~pytorch_ood.metrics.OODMetrics`. Each OOD dataset in ``ood_names`` is evaluated
        together with the ID test data.

        :param detector: detector instance or a sequence of detectors
        :param loader_kwargs: keyword arguments for :class:`torch.utils.data.DataLoader`, e.g.
            ``{"batch_size": 128}``. Without a ``batch_size``, samples are processed one at a time.
        :param device: device to move inputs and detectors to
        :param cache: keep the model outputs on the benchmark instance for later calls
        :param cache_dir: optional directory for file-backed caches. Requires ``cache_key``,
            otherwise a warning is issued and ``cache_dir`` is ignored.
        :param cache_key: name for the cached outputs, change it when the transforms or data change
        :return: list with one dictionary per OOD dataset (and per detector, if a sequence was
            given) containing the metrics of :class:`~pytorch_ood.metrics.OODMetrics` and the
            field ``Dataset`` with the name of the OOD dataset. For a sequence of detectors, each
            result also includes a ``Detector`` field with the detector class name.
        """
        detectors, many = self._normalize_detectors(detector)
        for current_detector in detectors:
            for name in ("model", "encoder", "backbone"):
                producer = getattr(current_detector, name, None)
                module = getattr(producer, "__self__", producer)
                if isinstance(module, torch.nn.Module) and module.training:
                    self._warn_once(
                        "train_mode",
                        f"The {name} of {type(current_detector).__name__} is in training mode, "
                        "which affects layers such as dropout and batch normalization. Call "
                        "eval() on it before evaluate() unless this is intended.",
                    )

        if loader_kwargs is None:
            loader_kwargs = {}

        if cache_dir is not None and cache_key is None:
            self._warn_once(
                "cache_dir_without_key",
                "File-backed benchmark caching was requested without a cache_key. "
                "The cache_dir is ignored.",
            )
            cache_dir = None
        elif cache_dir is not None:
            self._warn_once(
                f"cache_key_responsibility:{cache_key}",
                "The benchmark disk cache does not detect changes of the transforms, the data "
                "or model attributes other than parameters and buffers. Make sure your "
                "cache_key changes when they do.",
            )

        metrics = []
        identities: Dict = {}

        for dataset_name, dataset in zip(self.ood_names, self.test_sets()):
            loader = DataLoader(dataset=dataset, **loader_kwargs)
            local_cache = {}

            for current_detector in detectors:
                result = self._evaluate_single_detector(
                    detector=current_detector,
                    dataset_name=dataset_name,
                    data_loader=loader,
                    device=device,
                    cache=cache,
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    local_cache=local_cache,
                    identities=identities,
                )
                result.update({"Dataset": dataset_name})
                if many:
                    result.update({"Detector": type(current_detector).__name__})
                metrics.append(result)

        return metrics
