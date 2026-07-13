"""
Registry for pre-trained models.

Models are identified by a string key which, by convention, has the form
``<architecture>/<dataset>/<loss>/<seed>``, for example
``wrn-40-2/cifar10/logitnorm/s0``. The seed component is omitted for models
whose training seed is unknown (e.g. weights published by third parties).

Weights are downloaded (and cached) via ``torch.hub``, so the cache location can
be controlled with the ``TORCH_HOME`` environment variable. All checkpoints are
verified against their SHA-256 hash.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torchvision.transforms as tvt
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import Module

from pytorch_ood.utils import ToRGB

StateDictTransform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class Preprocessing(ABC):
    """
    Specification of the input preprocessing a model was trained with.

    Implementations are small, introspectable dataclasses; the actual transform
    is created with :meth:`build`.
    """

    @abstractmethod
    def build(self) -> Callable:
        """
        Create the transform that maps raw inputs to model inputs.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ImagePreprocessing(Preprocessing):
    """
    Preprocessing for image models: resize, conversion to RGB tensor, and
    normalization. The ``mean`` and ``std`` remain accessible as attributes,
    which is required by some detectors (e.g.
    :class:`ODIN <pytorch_ood.detector.ODIN>` needs ``std``).
    """

    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    size: Tuple[int, int] = (32, 32)

    def build(self) -> tvt.Compose:
        return tvt.Compose(
            [
                tvt.Resize(size=self.size),
                ToRGB(),
                tvt.ToTensor(),
                tvt.Normalize(mean=list(self.mean), std=list(self.std)),
            ]
        )


@dataclass(frozen=True)
class ModelEntry:
    """
    A single pre-trained model in the registry.
    """

    key: str
    """Unique identifier, by convention ``arch/dataset/loss[/seed]``."""

    arch_class: str
    """Dotted path of the architecture class, resolved lazily on load."""

    arch_kwargs: Dict[str, Any]
    """Constructor arguments, including ``num_classes`` where applicable."""

    url: str
    """Download URL of the ``.pt`` state dict."""

    sha256: str
    """Full SHA-256 digest of the checkpoint file."""

    # structured slots for filtering; by convention these make up the key
    arch: Optional[str] = None
    dataset: Optional[str] = None
    loss: Optional[str] = None
    seed: Optional[str] = None

    state_dict_transform: Optional[StateDictTransform] = None
    """Optional fixup applied to the raw state dict before loading."""

    preprocessing: Optional[Preprocessing] = None
    """Input preprocessing used during training; ``None`` if unknown."""

    metrics: Dict[str, float] = field(default_factory=dict)
    """Evaluation metrics, e.g. ``{"accuracy": 0.94}``."""

    source: Optional[str] = None
    """URL of the paper or upstream repository the weights originate from."""

    description: str = ""

    @property
    def file_name(self) -> str:
        """
        Cache filename. The trailing 16-hex-digit hash prefix enables
        integrity checking by ``torch.hub``.
        """
        return f"{self.key.replace('/', '-')}-{self.sha256[:16]}.pt"


def _strip_module_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Remove the ``module.`` prefix left behind by ``torch.nn.DataParallel``.
    """
    return {name.replace("module.", "", 1): param for name, param in state_dict.items()}


def _resolve_class(dotted_path: str) -> type:
    module_name, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _lookup(identifier: str) -> "ModelEntry":
    index = _index()

    if identifier in index:
        return index[identifier]

    # resolve as prefix, e.g. "wrn-40-2/cifar10/logitnorm" -> ".../s0"
    candidates = [key for key in index if key.startswith(identifier + "/")]
    if len(candidates) == 1:
        return index[candidates[0]]
    if len(candidates) > 1:
        raise ValueError(f"Identifier '{identifier}' is ambiguous. Candidates: {candidates}")

    raise ValueError(f"Unknown model '{identifier}'. Available models: {sorted(index)}")


def list_models(
    arch: Optional[str] = None,
    dataset: Optional[str] = None,
    loss: Optional[str] = None,
) -> List[str]:
    """
    List identifiers of all registered models, optionally filtered.

    :param arch: only include models with this architecture, e.g. ``wrn-40-2``
    :param dataset: only include models trained on this dataset, e.g. ``cifar10``
    :param loss: only include models trained with this method, e.g. ``oe``
    """
    entries = _ENTRIES
    if arch is not None:
        entries = [e for e in entries if e.arch == arch]
    if dataset is not None:
        entries = [e for e in entries if e.dataset == dataset]
    if loss is not None:
        entries = [e for e in entries if e.loss == loss]
    return sorted(e.key for e in entries)


def get_model_info(identifier: str) -> ModelEntry:
    """
    Look up the registry entry for a model.

    Exact matches win; an identifier without seed component (e.g.
    ``wrn-40-2/cifar10/logitnorm``) resolves if exactly one entry matches.

    :param identifier: model identifier
    :raises ValueError: if the identifier is unknown or ambiguous
    """
    return _lookup(identifier)


def load_model(identifier: str, map_location: str = "cpu") -> Module:
    """
    Download (or load from cache) a pre-trained model.

    Returns an instance of the concrete architecture class (see
    :attr:`ModelEntry.arch_class`), so architecture-specific attributes and
    methods, like intermediate layers, remain accessible. The model is
    returned in evaluation mode.

    :param identifier: model identifier, e.g. ``wrn-40-2/cifar10/logitnorm/s0``
    :param map_location: passed to ``torch.load``
    :raises ValueError: if the identifier is unknown or ambiguous
    """
    entry = _lookup(identifier)
    cls = _resolve_class(entry.arch_class)
    model: Module = cls(**entry.arch_kwargs)

    state_dict = load_state_dict_from_url(
        url=entry.url,
        map_location=map_location,
        file_name=entry.file_name,
        check_hash=True,
    )
    if entry.state_dict_transform is not None:
        state_dict = entry.state_dict_transform(state_dict)

    model.load_state_dict(state_dict)
    return model.eval()


def load_transform(identifier: str) -> Callable:
    """
    Create the input preprocessing for a pre-trained model.

    :param identifier: model identifier, e.g. ``wrn-40-2/cifar10/logitnorm/s0``
    :raises ValueError: if the identifier is unknown or the preprocessing of
        the model is not known
    """
    entry = _lookup(identifier)
    if entry.preprocessing is None:
        raise ValueError(f"Preprocessing for '{entry.key}' is not known")
    return entry.preprocessing.build()


_WRN = "pytorch_ood.model.WideResNet"
_WRN_40_2 = {"depth": 40, "widen_factor": 2, "drop_rate": 0.3}

_CIFAR_PREPROCESSING = ImagePreprocessing(
    mean=tuple(x / 255 for x in [125.3, 123.0, 113.9]),
    std=tuple(x / 255 for x in [63.0, 62.1, 66.7]),
)
_IMAGENET32_PREPROCESSING = ImagePreprocessing(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

_HF_BASE = "https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main"

_ENTRIES: List[ModelEntry] = [
    ModelEntry(
        key="wrn-40-2/cifar10/crossentropy",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="crossentropy",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=10, **_WRN_40_2),
        url="https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
        sha256="0b59e178577382a0cf5d1e079d1d019f6a33b0801b7d123a59e51b4b32972637",
        preprocessing=_CIFAR_PREPROCESSING,
        source="https://arxiv.org/abs/1610.02136",
        description="Baseline WideResNet-40-2 trained on CIFAR-10 with cross-entropy. "
        "Third-party weights, seed unknown.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/crossentropy",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="crossentropy",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=100, **_WRN_40_2),
        url="https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/pretrained/cifar100_wrn_pretrained_epoch_99.pt",
        sha256="3751f516b81ef9d27290f08a3de78d65a3771f477bd6a6caefa81b330dc34baf",
        preprocessing=_CIFAR_PREPROCESSING,
        source="https://arxiv.org/abs/1610.02136",
        description="Baseline WideResNet-40-2 trained on CIFAR-100 with cross-entropy. "
        "Third-party weights, seed unknown.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/oe",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="oe",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=10, **_WRN_40_2),
        url="https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots/oe_tune/cifar10_wrn_oe_tune_epoch_9.pt",
        sha256="b473f1c463c9fe16057647746907b68b6ade615f93ac4bdf005263ed69d542d0",
        preprocessing=_CIFAR_PREPROCESSING,
        source="https://arxiv.org/abs/1812.04606",
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Outlier Exposure using "
        "80 Million TinyImages. Third-party weights, seed unknown.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/oe",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="oe",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=100, **_WRN_40_2),
        url="https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots/oe_tune/cifar100_wrn_oe_tune_epoch_9.pt",
        sha256="1d4dd53f7605d66f64bcdf146ed9327ecbebbd5da42a9f7d70b9d6a28ee9bb51",
        preprocessing=_CIFAR_PREPROCESSING,
        source="https://arxiv.org/abs/1812.04606",
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Outlier Exposure using "
        "80 Million TinyImages. Third-party weights, seed unknown.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/energy/s1",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="energy",
        seed="s1",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=10, **_WRN_40_2),
        url="https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/energy_ft/cifar10_wrn_s1_energy_ft_epoch_9.pt",
        sha256="78a288517f9e9f3501c1e881aee8daa1d9c64a23805023fc013a28946274c54b",
        preprocessing=_CIFAR_PREPROCESSING,
        source="https://arxiv.org/abs/2010.03759",
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Energy Regularization "
        "using 80 Million TinyImages. Third-party weights.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/energy/s1",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="energy",
        seed="s1",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=100, **_WRN_40_2),
        url="https://github.com/wetliu/energy_ood/raw/master/CIFAR/snapshots/energy_ft/cifar100_wrn_s1_energy_ft_epoch_9.pt",
        sha256="4899110e41b8c96a01ec52e779e3336436da69db201f6162b2dce9a207f0f51f",
        preprocessing=_CIFAR_PREPROCESSING,
        source="https://arxiv.org/abs/2010.03759",
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Energy Regularization "
        "using 80 Million TinyImages. Third-party weights.",
    ),
    ModelEntry(
        key="wrn-40-2/imagenet32/crossentropy",
        arch="wrn-40-2",
        dataset="imagenet32",
        loss="crossentropy",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=1000, **_WRN_40_2),
        url="https://github.com/hendrycks/pre-training/raw/master/downsampled_train/snapshots/40_2/imagenet_wrn_baseline_epoch_99.pt",
        sha256="4a722ecf93e43a040b6b82c7c1e305fbf1c455a54c16e66fd77301291157bd5c",
        preprocessing=_IMAGENET32_PREPROCESSING,
        source="https://arxiv.org/abs/1901.09960",
        description="WideResNet-40-2 trained on a downscaled (32x32) version of the "
        "ImageNet. Third-party weights, seed unknown.",
    ),
    ModelEntry(
        key="wrn-40-2/imagenet32-nocifar/crossentropy",
        arch="wrn-40-2",
        dataset="imagenet32-nocifar",
        loss="crossentropy",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=1000, **_WRN_40_2),
        url="https://github.com/hendrycks/pre-training/raw/master/uncertainty/CIFAR/snapshots/imagenet/cifar10_excluded/imagenet_wrn_baseline_epoch_99.pt",
        sha256="6f0f46d44ddfc96f85b97ef04ca9d61e78279d27c80015dcc92aabd52b1773c6",
        state_dict_transform=_strip_module_prefix,
        preprocessing=_IMAGENET32_PREPROCESSING,
        source="https://arxiv.org/abs/1901.09960",
        description="WideResNet-40-2 trained on a downscaled (32x32) version of the "
        "ImageNet, excluding CIFAR-10 classes. Third-party weights, seed unknown.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/logitnorm/s0",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="logitnorm",
        seed="s0",
        arch_class=_WRN,
        arch_kwargs=dict(num_classes=10, **_WRN_40_2),
        url=f"{_HF_BASE}/wrn-40-2/cifar10/logitnorm/s0/model-bcd6754f4a4cfbd8.pt",
        sha256="bcd6754f4a4cfbd8b0cc28d432ef3fe63b715ab8bf2776dec42059ae445b7419",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467), std=(0.247, 0.2435, 0.2616)
        ),
        metrics={"final_accuracy": 0.943, "best_accuracy": 0.9436},
        source="https://arxiv.org/abs/2205.09310",
        description="WideResNet-40-2 trained on CIFAR-10 with LogitNorm (t=0.04) for "
        "100 epochs with SGD.",
    ),
]

_INDEX_CACHE: Optional[Dict[str, ModelEntry]] = None


def _index() -> Dict[str, ModelEntry]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        index = {}
        for entry in _ENTRIES:
            assert entry.key not in index, f"Duplicate registry key: {entry.key}"
            index[entry.key] = entry
        _INDEX_CACHE = index
    return _INDEX_CACHE
