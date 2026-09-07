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
    crop_size: Optional[Tuple[int, int]] = None
    """If set, images are resized to ``size`` and then center-cropped to
    ``crop_size`` (the standard ImageNet-style eval pipeline), instead of being
    resized to ``size`` directly."""

    def build(self) -> tvt.Compose:
        transforms = [tvt.Resize(size=self.size)]
        if self.crop_size is not None:
            transforms.append(tvt.CenterCrop(size=self.crop_size))
        transforms += [
            ToRGB(),
            tvt.ToTensor(),
            tvt.Normalize(mean=list(self.mean), std=list(self.std)),
        ]
        return tvt.Compose(transforms)


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
    ModelEntry(
        key="resnet-18/imagenet200/crossentropy/s0",
        arch="resnet-18",
        dataset="imagenet200",
        loss="crossentropy",
        seed="s0",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/crossentropy/s0/model-f4807cb4ea6828f3.pt",
        sha256="f4807cb4ea6828f3ebfa21bac46e04e340b242b90d46f2e9ea4cd70bcf300e76",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8500000238418579, "best_accuracy": 0.8579999804496765},
        description="ResNet-18 trained on ImageNet-200 with cross-entropy for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/crossentropy/s1",
        arch="resnet-18",
        dataset="imagenet200",
        loss="crossentropy",
        seed="s1",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/crossentropy/s1/model-64022f83262be54a.pt",
        sha256="64022f83262be54a74a58ee8f2d64f69a515a7f64f7b2b7ab2d9ddd000e2534d",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8450000286102295, "best_accuracy": 0.8489999771118164},
        description="ResNet-18 trained on ImageNet-200 with cross-entropy for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/crossentropy/s2",
        arch="resnet-18",
        dataset="imagenet200",
        loss="crossentropy",
        seed="s2",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/crossentropy/s2/model-ac9415e147eb6137.pt",
        sha256="ac9415e147eb6137516c6db7672b2899a5a0f2e8dbcdcbc7f4a8eaa37d5454d1",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8420000076293945, "best_accuracy": 0.8479999899864197},
        description="ResNet-18 trained on ImageNet-200 with cross-entropy for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/entropic/s0",
        arch="resnet-18",
        dataset="imagenet200",
        loss="entropic",
        seed="s0",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/entropic/s0/model-93fef586a085e884.pt",
        sha256="93fef586a085e884bad7048d8195d5a5ba93f053f401ac30eeefd871dd31f3cc",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8489999771118164, "best_accuracy": 0.8489999771118164},
        description="ResNet-18 trained on ImageNet-200 with EntropicOpenSetLoss (outlier exposure using ImageNet-800) for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/entropic/s1",
        arch="resnet-18",
        dataset="imagenet200",
        loss="entropic",
        seed="s1",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/entropic/s1/model-5db3425a88b59bf9.pt",
        sha256="5db3425a88b59bf95c3b154f1b87df2fad77440adddfb40d679e96773959c494",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.847000002861023, "best_accuracy": 0.847000002861023},
        description="ResNet-18 trained on ImageNet-200 with EntropicOpenSetLoss (outlier exposure using ImageNet-800) for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/entropic/s2",
        arch="resnet-18",
        dataset="imagenet200",
        loss="entropic",
        seed="s2",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/entropic/s2/model-8d9119218862acf9.pt",
        sha256="8d9119218862acf978013a2cde74bd1799efdeb9e590b6aceb9a2b7b70389da5",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8370000123977661, "best_accuracy": 0.8460000157356262},
        description="ResNet-18 trained on ImageNet-200 with EntropicOpenSetLoss (outlier exposure using ImageNet-800) for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/oe/s0",
        arch="resnet-18",
        dataset="imagenet200",
        loss="oe",
        seed="s0",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/oe/s0/model-5798c74848753fee.pt",
        sha256="5798c74848753feef58215813c5f38a941459e58aa59637952f7e0bf42ce5b23",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8579999804496765, "best_accuracy": 0.8579999804496765},
        description="ResNet-18 fine-tuned on ImageNet-200 with Outlier Exposure using ImageNet-800 for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s0.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/oe/s1",
        arch="resnet-18",
        dataset="imagenet200",
        loss="oe",
        seed="s1",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/oe/s1/model-4c1b951acf950340.pt",
        sha256="4c1b951acf9503406f75484ba2cd5ef42c75bcca39588d0b1c186a3dd288b4cc",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8460000157356262, "best_accuracy": 0.8500000238418579},
        description="ResNet-18 fine-tuned on ImageNet-200 with Outlier Exposure using ImageNet-800 for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s1.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/oe/s2",
        arch="resnet-18",
        dataset="imagenet200",
        loss="oe",
        seed="s2",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/oe/s2/model-da72bbbd08f4c707.pt",
        sha256="da72bbbd08f4c70778a842ddf37caa7da05ee13c4c5cca92c896457830a646fb",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8389999866485596, "best_accuracy": 0.8410000205039978},
        description="ResNet-18 fine-tuned on ImageNet-200 with Outlier Exposure using ImageNet-800 for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s2.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/energy/s0",
        arch="resnet-18",
        dataset="imagenet200",
        loss="energy",
        seed="s0",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/energy/s0/model-e5694bf7d2bc1627.pt",
        sha256="e5694bf7d2bc162714e8c1accf01a3793977586e4905db6c69b27754a8ea0eb1",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8450000286102295, "best_accuracy": 0.8450000286102295},
        description="ResNet-18 fine-tuned on ImageNet-200 with EnergyRegularizedLoss using ImageNet-800 for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s0.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/energy/s1",
        arch="resnet-18",
        dataset="imagenet200",
        loss="energy",
        seed="s1",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/energy/s1/model-d0792e4d01e8688f.pt",
        sha256="d0792e4d01e8688f43ce739962b3a870df3931aa27d6c54d7eeb2e08cd7829d8",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8460000157356262, "best_accuracy": 0.847000002861023},
        description="ResNet-18 fine-tuned on ImageNet-200 with EnergyRegularizedLoss using ImageNet-800 for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s1.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/energy/s2",
        arch="resnet-18",
        dataset="imagenet200",
        loss="energy",
        seed="s2",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/energy/s2/model-cdd46dc500a0d800.pt",
        sha256="cdd46dc500a0d80070d0784031c601e90abd3dbf0c7098a0ea10c649d3f0068c",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8259999752044678, "best_accuracy": 0.8270000219345093},
        description="ResNet-18 fine-tuned on ImageNet-200 with EnergyRegularizedLoss using ImageNet-800 for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s2.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/logitnorm/s0",
        arch="resnet-18",
        dataset="imagenet200",
        loss="logitnorm",
        seed="s0",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/logitnorm/s0/model-08eed212862970d4.pt",
        sha256="08eed212862970d40145cb78443cbacdcc52e93aeb2c49f17c25c0a4f6b544ea",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.847000002861023, "best_accuracy": 0.8569999933242798},
        description="ResNet-18 trained on ImageNet-200 with LogitNorm for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/logitnorm/s1",
        arch="resnet-18",
        dataset="imagenet200",
        loss="logitnorm",
        seed="s1",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/logitnorm/s1/model-f252ad1c8d33caa4.pt",
        sha256="f252ad1c8d33caa4a3e33d8cdfa3850cb3f74d5cd440795d54142b026db473f0",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8460000157356262, "best_accuracy": 0.847000002861023},
        description="ResNet-18 trained on ImageNet-200 with LogitNorm for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/logitnorm/s2",
        arch="resnet-18",
        dataset="imagenet200",
        loss="logitnorm",
        seed="s2",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/logitnorm/s2/model-ce53f305ec80ee41.pt",
        sha256="ce53f305ec80ee414aee0c9debf0cc597d87f290d319e53e63251af78f05dd93",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.847000002861023, "best_accuracy": 0.8529999852180481},
        description="ResNet-18 trained on ImageNet-200 with LogitNorm for 90 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/crossentropy/s0",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="crossentropy",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/crossentropy/s0/model-4b5b3ed2f448b40c.pt",
        sha256="4b5b3ed2f448b40cc48abe05221b4af39e351b04ce215edf2c0ea7a113593d2f",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7556999921798706, "best_accuracy": 0.7576000094413757},
        description="WideResNet-40-2 trained on CIFAR-100 with cross-entropy for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/crossentropy/s1",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="crossentropy",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/crossentropy/s1/model-1b8e15072dc0d887.pt",
        sha256="1b8e15072dc0d88776dc59c23192628354b958fa22bdc21073664d60d862d685",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7570000290870667, "best_accuracy": 0.7605000138282776},
        description="WideResNet-40-2 trained on CIFAR-100 with cross-entropy for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/crossentropy/s2",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="crossentropy",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/crossentropy/s2/model-f1a77c51f747d58e.pt",
        sha256="f1a77c51f747d58e16bc4a7a04d8a6b7f65d3a7c694e63e5b6480fd08a8f423e",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7610999941825867, "best_accuracy": 0.761900007724762},
        description="WideResNet-40-2 trained on CIFAR-100 with cross-entropy for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/entropic/s0",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="entropic",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/entropic/s0/model-cbabacfd3c396d11.pt",
        sha256="cbabacfd3c396d11d115afcdbd242dd863fb5de863907a3cf5d15e6630a6eeb6",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7584999799728394, "best_accuracy": 0.7585999965667725},
        description="WideResNet-40-2 trained on CIFAR-100 with EntropicOpenSetLoss (outlier exposure using 80 Million TinyImages) for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/entropic/s1",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="entropic",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/entropic/s1/model-bd96368e7713e60b.pt",
        sha256="bd96368e7713e60b9f312767e9a883e097cc9e17ce3564ff7c8a5c23afd6a99b",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.757099986076355, "best_accuracy": 0.7590000033378601},
        description="WideResNet-40-2 trained on CIFAR-100 with EntropicOpenSetLoss (outlier exposure using 80 Million TinyImages) for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/entropic/s2",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="entropic",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/entropic/s2/model-f572bdb2685bcede.pt",
        sha256="f572bdb2685bcede584ad3ad110ab77310983c4b99252f2bd457df5e1ff86593",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7627999782562256, "best_accuracy": 0.7642999887466431},
        description="WideResNet-40-2 trained on CIFAR-100 with EntropicOpenSetLoss (outlier exposure using 80 Million TinyImages) for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/logitnorm/s0",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="logitnorm",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/logitnorm/s0/model-cfd67c89d658c2c9.pt",
        sha256="cfd67c89d658c2c9b04b26b8ec82a1154ebb7e765fb0c4ba42c73b95c56416e6",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7587000131607056, "best_accuracy": 0.7598000168800354},
        description="WideResNet-40-2 trained on CIFAR-100 with LogitNorm for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/logitnorm/s1",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="logitnorm",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/logitnorm/s1/model-34c3830934542889.pt",
        sha256="34c383093454288984d09ba998852f394eb3eb511727679a7e1a05bc65184e65",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7634999752044678, "best_accuracy": 0.7656000256538391},
        description="WideResNet-40-2 trained on CIFAR-100 with LogitNorm for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/logitnorm/s2",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="logitnorm",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/logitnorm/s2/model-663f69475582ae9f.pt",
        sha256="663f69475582ae9f034755aa123e995cbd9b82f29a74fe2e4a5f62276ac60182",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7638999819755554, "best_accuracy": 0.7642999887466431},
        description="WideResNet-40-2 trained on CIFAR-100 with LogitNorm for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/crossentropy/s1",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="crossentropy",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/crossentropy/s1/model-b1b7465567ee2494.pt",
        sha256="b1b7465567ee249481b8ff631f9d96fcc80d4c27783dc117aa2020888607cdf7",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9480000138282776, "best_accuracy": 0.9484000205993652},
        description="WideResNet-40-2 trained on CIFAR-10 with cross-entropy for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/crossentropy/s2",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="crossentropy",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/crossentropy/s2/model-9c18ac23901a26b7.pt",
        sha256="9c18ac23901a26b7787de79042051ec90a75e8036e5b2c7858c2db8ab437816b",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9484000205993652, "best_accuracy": 0.9484000205993652},
        description="WideResNet-40-2 trained on CIFAR-10 with cross-entropy for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/entropic/s0",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="entropic",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/entropic/s0/model-481c83f11ccb4b2a.pt",
        sha256="481c83f11ccb4b2ae1b99e7480c62582c93d08fc15c368f8708379b6a60f08da",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9534000158309937, "best_accuracy": 0.9539999961853027},
        description="WideResNet-40-2 trained on CIFAR-10 with EntropicOpenSetLoss (outlier exposure using 80 Million TinyImages) for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/entropic/s1",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="entropic",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/entropic/s1/model-653ef4ffd71c6804.pt",
        sha256="653ef4ffd71c68048527f1d7bcd8272c3c66a8c298acfa3ebe9997c388d4946a",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.953499972820282, "best_accuracy": 0.9541000127792358},
        description="WideResNet-40-2 trained on CIFAR-10 with EntropicOpenSetLoss (outlier exposure using 80 Million TinyImages) for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/entropic/s2",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="entropic",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/entropic/s2/model-cb8865fedf8ed74e.pt",
        sha256="cb8865fedf8ed74eff2fbbb72c07d9e03e6c57c1cdba68bb4b3f788ee4b49b00",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9534000158309937, "best_accuracy": 0.9541000127792358},
        description="WideResNet-40-2 trained on CIFAR-10 with EntropicOpenSetLoss (outlier exposure using 80 Million TinyImages) for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/logitnorm/s1",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="logitnorm",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/logitnorm/s1/model-7d932b6fe7a6bb43.pt",
        sha256="7d932b6fe7a6bb435d50345b88234286ab5d859e02cb2451146f3e4d2f9605ba",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9478999972343445, "best_accuracy": 0.9478999972343445},
        description="WideResNet-40-2 trained on CIFAR-10 with LogitNorm for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/logitnorm/s2",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="logitnorm",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/logitnorm/s2/model-7de464bd4ce3c4bf.pt",
        sha256="7de464bd4ce3c4bf6efaa1dddddb23e38bed99653036a38ef02bcb50cff214f6",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9434999823570251, "best_accuracy": 0.9435999989509583},
        description="WideResNet-40-2 trained on CIFAR-10 with LogitNorm for 100 epochs with SGD.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/oe/s0",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="oe",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/oe/s0/model-4b018f801843365c.pt",
        sha256="4b018f801843365c8ba578d20caa97be2f995eb7eb93c5909f1c0273bc7b6022",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9449999928474426, "best_accuracy": 0.9453999996185303},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Outlier Exposure using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar10/crossentropy.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/oe/s1",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="oe",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/oe/s1/model-71896c06742c88c7.pt",
        sha256="71896c06742c88c7135d2869682ee1fe190d3b5e1f63129c65f1fdd338908e2a",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9413999915122986, "best_accuracy": 0.9416999816894531},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Outlier Exposure using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar10/crossentropy/s1.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/oe/s2",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="oe",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/oe/s2/model-0e08d73c2cc42595.pt",
        sha256="0e08d73c2cc4259562f5bcd09abfb33c6b53c10e3010b807434d68da98165bb5",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9455999732017517, "best_accuracy": 0.9455999732017517},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Outlier Exposure using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar10/crossentropy/s2.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/oe/s0",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="oe",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/oe/s0/model-f713cea10b512718.pt",
        sha256="f713cea10b512718f1dc8772f54b3220a827a23114713bcc9f50d95d1460d920",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7457000017166138, "best_accuracy": 0.7458000183105469},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Outlier Exposure using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar100/crossentropy/s0.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/oe/s1",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="oe",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/oe/s1/model-4d9c5eaac646477b.pt",
        sha256="4d9c5eaac646477b706f006337b4a0f7c49a2be4bc064f230dd7766229851e89",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7516999840736389, "best_accuracy": 0.7516999840736389},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Outlier Exposure using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar100/crossentropy/s1.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/oe/s2",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="oe",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/oe/s2/model-51ec426d53fe2a27.pt",
        sha256="51ec426d53fe2a2747bfa03301b905029166eaa09d38061158905265cf5fe05a",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7494999766349792, "best_accuracy": 0.7501000165939331},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Outlier Exposure using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar100/crossentropy/s2.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/energy/s0",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="energy",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/energy/s0/model-d798f066e7912d4a.pt",
        sha256="d798f066e7912d4a267c387121f2dcaa4b59611a2559269462972cb9eda4e812",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9524999856948853, "best_accuracy": 0.9524999856948853},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Energy Regularization using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar10/crossentropy.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/energy/s2",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="energy",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/energy/s2/model-92f4f18cb4dfad21.pt",
        sha256="92f4f18cb4dfad21e22a4bf43d1674b952ffe60885fe3a0a7782929007001559",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.949400007724762, "best_accuracy": 0.9501000046730042},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with Energy Regularization using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar10/crossentropy/s2.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/energy/s0",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="energy",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/energy/s0/model-ebd0fcc793f5a22f.pt",
        sha256="ebd0fcc793f5a22fbf20f1d29638138b36c69eebcf64f9cf96aa2c945259e420",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7558000087738037, "best_accuracy": 0.7563999891281128},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Energy Regularization using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar100/crossentropy/s0.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/energy/s2",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="energy",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/energy/s2/model-50d7d6802391a24f.pt",
        sha256="50d7d6802391a24f2f2aa32603f07d97a9761b6aadeb21e74f50857a5873ccbb",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7616999745368958, "best_accuracy": 0.7616999745368958},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with Energy Regularization using 300K TinyImages for 10 epochs (lr=0.001), initialized from wrn-40-2/cifar100/crossentropy/s2.",
    ),
    ModelEntry(
        key="resnet-50/imagenet1k/crossentropy",
        arch="resnet-50",
        dataset="imagenet1k",
        loss="crossentropy",
        arch_class="pytorch_ood.model.ResNet50",
        arch_kwargs={"num_classes": 1000},
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        sha256="11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(232, 232),
            crop_size=(224, 224),
        ),
        source="https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html",
        description="Standard torchvision ResNet-50 pretrained on full ImageNet-1K "
        "(IMAGENET1K_V2 weights, 80.858% top-1 accuracy per torchvision's own "
        "reported metrics). Third-party weights, seed unknown; not trained by "
        "pytorch-ood.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/vos/s0",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="vos",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/vos/s0/model-8a70f61ad4ce2c6f.pt",
        sha256="8a70f61ad4ce2c6fdebf15ce5408a28d2880dfb890ed45614e906cc6a97e18f4",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9466000199317932, "best_accuracy": 0.9466000199317932},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from 300K TinyImages are used instead) for 10 epochs (lr=0.1), initialized from wrn-40-2/cifar10/crossentropy.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/vos/s1",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="vos",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/vos/s1/model-3bc58d14199aabd8.pt",
        sha256="3bc58d14199aabd838484318a36c58e0d0b2cf1e3e7556a3ce20dd538b4d0b41",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9469000101089478, "best_accuracy": 0.9469000101089478},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from 300K TinyImages are used instead) for 10 epochs (lr=0.1), initialized from wrn-40-2/cifar10/crossentropy/s1.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar10/vos/s2",
        arch="wrn-40-2",
        dataset="cifar10",
        loss="vos",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 10, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar10/vos/s2/model-cda47e9257f78549.pt",
        sha256="cda47e9257f785499473ebc9ff7a6b238bedc6d7c425a207408b21aaf2c1ec6b",
        preprocessing=ImagePreprocessing(
            mean=(0.4913, 0.4823, 0.4467),
            std=(0.247, 0.2435, 0.2616),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.9474999904632568, "best_accuracy": 0.9474999904632568},
        description="WideResNet-40-2 fine-tuned on CIFAR-10 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from 300K TinyImages are used instead) for 10 epochs (lr=0.1), initialized from wrn-40-2/cifar10/crossentropy/s2.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/vos/s0",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="vos",
        seed="s0",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/vos/s0/model-49e6b2e898c7a486.pt",
        sha256="49e6b2e898c7a486300273ee6bea7209ccfc11b6642de130319c60d9e6f3ff81",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7581999897956848, "best_accuracy": 0.7581999897956848},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from 300K TinyImages are used instead) for 10 epochs (lr=0.1), initialized from wrn-40-2/cifar100/crossentropy/s0.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/vos/s1",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="vos",
        seed="s1",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/vos/s1/model-7c9fd1931e5b1b62.pt",
        sha256="7c9fd1931e5b1b62ba098fcafab550b93ff189ceae3f37f5bccbd27b95c445f7",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7620999813079834, "best_accuracy": 0.7620999813079834},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from 300K TinyImages are used instead) for 10 epochs (lr=0.1), initialized from wrn-40-2/cifar100/crossentropy/s1.",
    ),
    ModelEntry(
        key="wrn-40-2/cifar100/vos/s2",
        arch="wrn-40-2",
        dataset="cifar100",
        loss="vos",
        seed="s2",
        arch_class="pytorch_ood.model.WideResNet",
        arch_kwargs={"num_classes": 100, "depth": 40, "widen_factor": 2, "drop_rate": 0.3},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/wrn-40-2/cifar100/vos/s2/model-4eaffc7283aa6993.pt",
        sha256="4eaffc7283aa6993cd86e378fbf877d14c1c1f07e8b8782bfe94eb0acbb44aca",
        preprocessing=ImagePreprocessing(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
            size=(32, 32),
            crop_size=None,
        ),
        metrics={"final_accuracy": 0.7570000290870667, "best_accuracy": 0.7570000290870667},
        description="WideResNet-40-2 fine-tuned on CIFAR-100 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from 300K TinyImages are used instead) for 10 epochs (lr=0.1), initialized from wrn-40-2/cifar100/crossentropy/s2.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/vos/s0",
        arch="resnet-18",
        dataset="imagenet200",
        loss="vos",
        seed="s0",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/vos/s0/model-6125bfafe3b6c92d.pt",
        sha256="6125bfafe3b6c92d85b32007d83d247de54d72948fe68781eefa2e4d3d0739a1",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8539999723434448, "best_accuracy": 0.8550000190734863},
        description="ResNet-18 fine-tuned on ImageNet-200 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from ImageNet-800 are used instead) for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s0.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/vos/s1",
        arch="resnet-18",
        dataset="imagenet200",
        loss="vos",
        seed="s1",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/vos/s1/model-284911365e826187.pt",
        sha256="284911365e826187b2f77ce569284ff9545503863e8fd84945e2956b6ca7526c",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8500000238418579, "best_accuracy": 0.8529999852180481},
        description="ResNet-18 fine-tuned on ImageNet-200 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from ImageNet-800 are used instead) for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s1.",
    ),
    ModelEntry(
        key="resnet-18/imagenet200/vos/s2",
        arch="resnet-18",
        dataset="imagenet200",
        loss="vos",
        seed="s2",
        arch_class="pytorch_ood.model.ResNet18",
        arch_kwargs={"num_classes": 200, "weights": None},
        url="https://huggingface.co/kkirchheim/pytorch-ood-models/resolve/main/resnet-18/imagenet200/vos/s2/model-8f9642c7f6d66d39.pt",
        sha256="8f9642c7f6d66d39658e36bcc898c25e55cf24dd04434fdc27860d86709ad58f",
        preprocessing=ImagePreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            size=(256, 256),
            crop_size=(224, 224),
        ),
        metrics={"final_accuracy": 0.8389999866485596, "best_accuracy": 0.843999981880188},
        description="ResNet-18 fine-tuned on ImageNet-200 with VOS (energy-based regularization from *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, without the virtual-outlier sampling -- real outliers from ImageNet-800 are used instead) for 10 epochs (lr=0.001), initialized from resnet-18/imagenet200/crossentropy/s2.",
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
