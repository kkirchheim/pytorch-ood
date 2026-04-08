import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_ood.api import Detector, FeaturesDetector, LogitsDetector
from pytorch_ood.detector.mahalanobis import Mahalanobis
from pytorch_ood.utils import OODMetrics, TensorBuffer

_CACHE_VERSION = 1


def _sanitize_cache_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "cache"


class Benchmark(ABC):
    """
    Base class for Benchmarks
    """

    @abstractmethod
    def train_set(self) -> Dataset:
        """
        Training dataset
        """

    @abstractmethod
    def test_sets(self, known=True, unknown=True) -> List[Dataset]:
        """
        List of the different test datasets.
        If known and unknown are true, each dataset contains ID and OOD data.

        :param known: include ID
        :param unknown: include OOD
        """
        pass

    def _ensure_cache_state(self):
        if not hasattr(self, "_representation_cache"):
            self._representation_cache = {}

        if not hasattr(self, "_cache_warnings_shown"):
            self._cache_warnings_shown = set()

    def _warn_once(self, key: str, message: str) -> None:
        self._ensure_cache_state()
        if key in self._cache_warnings_shown:
            return

        warnings.warn(message, UserWarning, stacklevel=3)
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
        for attr in ("model", "encoder", "backbone"):
            if hasattr(detector, attr):
                producer = getattr(detector, attr)
                if producer is not None:
                    return producer
        return None

    @staticmethod
    def _producer_token(producer) -> str:
        if producer is None:
            return "none"
        return f"{producer.__class__.__module__}.{producer.__class__.__qualname__}"

    def _memory_cache_key(
        self,
        split: str,
        dataset_name: str,
        representation: str,
        producer,
        cache_key: Optional[str],
    ) -> Tuple[str, str, str, int, Optional[str]]:
        return split, dataset_name, representation, id(producer), cache_key

    def _cache_file_path(
        self,
        cache_dir: str,
        split: str,
        dataset_name: str,
        representation: str,
        producer,
        cache_key: str,
    ) -> Path:
        producer_token = _sanitize_cache_token(self._producer_token(producer))
        dataset_token = _sanitize_cache_token(dataset_name)
        split_token = _sanitize_cache_token(split)
        representation_token = _sanitize_cache_token(representation)
        cache_key_token = _sanitize_cache_token(cache_key)
        filename = (
            f"{cache_key_token}_{split_token}_{dataset_token}_"
            f"{representation_token}_{producer_token}.pt"
        )
        return Path(cache_dir) / filename

    def _load_disk_cache(
        self,
        cache_dir: Optional[str],
        split: str,
        dataset_name: str,
        representation: str,
        producer,
        cache_key: Optional[str],
    ) -> Optional[Dict]:
        if cache_dir is None or cache_key is None:
            return None

        path = self._cache_file_path(
            cache_dir=cache_dir,
            split=split,
            dataset_name=dataset_name,
            representation=representation,
            producer=producer,
            cache_key=cache_key,
        )

        if not path.exists():
            return None

        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        expected = {
            "cache_version": _CACHE_VERSION,
            "split": split,
            "dataset_name": dataset_name,
            "representation": representation,
            "cache_key": cache_key,
            "producer_token": self._producer_token(producer),
        }

        for key, value in expected.items():
            if metadata.get(key) != value:
                return None

        return payload

    def _save_disk_cache(
        self,
        payload: Dict,
        cache_dir: Optional[str],
        split: str,
        dataset_name: str,
        representation: str,
        producer,
        cache_key: Optional[str],
    ) -> None:
        if cache_dir is None or cache_key is None:
            return

        path = self._cache_file_path(
            cache_dir=cache_dir,
            split=split,
            dataset_name=dataset_name,
            representation=representation,
            producer=producer,
            cache_key=cache_key,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

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
        persist_memory: bool,
        cache_dir: Optional[str],
        cache_key: Optional[str],
        local_cache: Dict,
    ) -> Dict:
        self._ensure_cache_state()
        memory_key = self._memory_cache_key(
            split=split,
            dataset_name=dataset_name,
            representation=representation,
            producer=producer,
            cache_key=cache_key,
        )

        if memory_key in local_cache:
            return local_cache[memory_key]

        if persist_memory and memory_key in self._representation_cache:
            payload = self._representation_cache[memory_key]
            local_cache[memory_key] = payload
            return payload

        payload = self._load_disk_cache(
            cache_dir=cache_dir,
            split=split,
            dataset_name=dataset_name,
            representation=representation,
            producer=producer,
            cache_key=cache_key,
        )
        if payload is None:
            data = self._extract_representation(
                data_loader=data_loader,
                producer=producer,
                device=device,
                representation=representation,
            )
            payload = {
                "metadata": {
                    "cache_version": _CACHE_VERSION,
                    "split": split,
                    "dataset_name": dataset_name,
                    "representation": representation,
                    "cache_key": cache_key,
                    "producer_token": self._producer_token(producer),
                    "num_samples": int(data["label"].shape[0]),
                },
                "data": data,
            }
            self._save_disk_cache(
                payload=payload,
                cache_dir=cache_dir,
                split=split,
                dataset_name=dataset_name,
                representation=representation,
                producer=producer,
                cache_key=cache_key,
            )

        if persist_memory:
            self._representation_cache[memory_key] = payload
        local_cache[memory_key] = payload
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

        if isinstance(detector, Mahalanobis) and detector.eps > 0:
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
        metrics.update(scores, labels.to(scores.device))
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
        persist_memory: bool,
        cache_dir: Optional[str],
        cache_key: Optional[str],
        local_cache: Dict,
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
                persist_memory=persist_memory,
                cache_dir=cache_dir,
                cache_key=cache_key,
                local_cache=local_cache,
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
                    persist_memory=persist_memory,
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    local_cache=local_cache,
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
    ) -> List[Dict]:
        ...

    @overload
    def evaluate(
        self,
        detector: Sequence[Detector],
        loader_kwargs: Optional[Dict] = None,
        device: str = "cpu",
        cache: bool = False,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> List[Dict]:
        ...

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
        together, this method can reuse cached intermediate representations
        instead of recomputing model outputs for every detector. If ``cache=True``,
        those representations are also kept on the benchmark instance and reused
        across later ``evaluate(...)`` calls. If ``cache_dir`` is given, cached
        tensors are additionally persisted to disk.

        Disk-backed cache reuse is keyed only by user-provided ``cache_key`` and
        lightweight metadata, so cache correctness is the caller's responsibility.

        :param detector: detector instance or a sequence of detectors
        :param loader_kwargs: keyword arguments forwarded to the data loader
        :param device: device to move inputs and detectors to
        :param cache: keep cached representations on the benchmark instance
        :param cache_dir: optional directory for file-backed caches
        :param cache_key: user-supplied cache key used for disk cache reuse
        :return: benchmark results. For multiple detectors, each result includes
            a ``Detector`` field with the detector class name.
        """
        detectors, many = self._normalize_detectors(detector)

        if loader_kwargs is None:
            loader_kwargs = {}

        persist_memory = cache or cache_dir is not None
        if cache_dir is not None and cache_key is None:
            self._warn_once(
                "cache_dir_without_key",
                "File-backed benchmark caching was requested without a cache_key. "
                "Disk cache reuse is disabled; only in-memory caching will be used.",
            )
            cache_dir = None
        elif cache_dir is not None:
            self._warn_once(
                f"cache_key_responsibility:{cache_key}",
                "Benchmark cache reuse is keyed by the user-supplied cache_key and "
                "lightweight metadata only. Make sure your cache_key changes when "
                "the model, weights, or transforms change.",
            )

        metrics = []

        for dataset_name, dataset in zip(self.ood_names, self.test_sets()):
            loader = DataLoader(dataset=dataset, **loader_kwargs)
            local_cache = {}

            for current_detector in detectors:
                result = self._evaluate_single_detector(
                    detector=current_detector,
                    dataset_name=dataset_name,
                    data_loader=loader,
                    device=device,
                    persist_memory=persist_memory,
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    local_cache=local_cache,
                )
                result.update({"Dataset": dataset_name})
                if many:
                    result.update({"Detector": type(current_detector).__name__})
                metrics.append(result)

        return metrics
