"""
One working instance of every detector, for tests that run over all of them.

Each :class:`DetectorCase` creates a small model in ``eval()`` mode and wraps it in a detector, for
inputs of ``input_shape`` per sample. Tests that check something for every detector parametrize
over :data:`DETECTOR_CASES`.

The builders are deterministic: models, fit data and other random parts are drawn with fixed seeds,
so every run and every test order checks the same weights and data, and the global RNG is left as
it was.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import Detector
from pytorch_ood.detector import (
    ASH,
    DICE,
    GEN,
    GMM,
    KNN,
    LTS,
    MCD,
    MCM,
    NACUE,
    NCI,
    ODIN,
    PNML,
    RMD,
    SCALE,
    SHE,
    VRA,
    EnergyBased,
    Entropy,
    GradNorm,
    GradUncertainty,
    Gram,
    KLMatching,
    Mahalanobis,
    MahalanobisODIN,
    MaxLogit,
    MaxSoftmax,
    MultiMahalanobis,
    NNGuide,
    OpenMax,
    RankFeat,
    ReAct,
    Residual,
    TemperatureScaling,
    ViM,
    WeightedEBO,
    fDBD,
)
from tests.helpers.model import ClassificationModel

VECTOR = (10,)
IMAGE = (3, 8, 8)

#: seed of the toy models and of other random parts of a detector
MODEL_SEED = 0
#: seed of the default fit data; the test inputs draw from the ``seed`` fixture (seed 0), so this
#: one differs and the inputs are not training samples
FIT_SEED = 1


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@contextmanager
def _seeded(seed: int = MODEL_SEED) -> Iterator[None]:
    """Seeds the global CPU RNG, which ``nn.Module`` initialization draws from, and restores its
    state afterwards."""
    with torch.random.fork_rng(devices=[]):
        # not torch.manual_seed: it would also reseed CUDA, which fork_rng(devices=[]) does not
        # restore
        torch.random.default_generator.manual_seed(seed)
        yield


@dataclass(frozen=True)
class DetectorCase:
    name: str
    #: creates a fresh model in ``eval()`` mode
    model: Callable[[], nn.Module]
    #: wraps a model from :attr:`model` in the detector
    detector: Callable[[nn.Module], Detector]
    input_shape: Tuple[int, ...]
    #: replaces the default fit data of :func:`make_loader` for detectors that need special data
    fit_loader: Optional[Callable[[], DataLoader]] = None

    def __str__(self) -> str:
        return self.name

    def build(self) -> Detector:
        return self.detector(self.model())

    def make_fit_loader(self, **kwargs) -> DataLoader:
        """:attr:`fit_loader` if set, otherwise :func:`make_loader` with ``kwargs``."""
        return self.fit_loader() if self.fit_loader else make_loader(self.input_shape, **kwargs)


def make_loader(
    input_shape: Tuple[int, ...],
    n_samples: int = 24,
    batch_size: int = 6,
    n_classes: int = 3,
    seed: int = FIT_SEED,
) -> DataLoader:
    """Random in-distribution inputs with labels ``0 .. n_classes - 1``, the same for the same
    ``seed``."""
    x = torch.randn(n_samples, *input_shape, generator=_generator(seed))
    y = torch.arange(n_samples) % n_classes
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class TinyConvDetectorModel(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=8, num_outputs=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_channels, num_outputs)

        # aliases used by several detectors
        self.block1 = self.relu
        self.block2 = nn.Identity()
        self.block3 = nn.Identity()
        self.bn1 = nn.Identity()

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv1(x))

    def forward_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(self.pool(x)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(self.feature_maps(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def _classifier(**kwargs) -> ClassificationModel:
    with _seeded():
        return ClassificationModel(**kwargs).eval()


def _gradient_classifier() -> ClassificationModel:
    # only the last layer takes gradients, as the gradient-based detectors are usually configured
    model = _classifier()
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)
    return model


def _conv() -> TinyConvDetectorModel:
    with _seeded():
        return TinyConvDetectorModel().eval()


def _she_classifier() -> ClassificationModel:
    # features are the first three inputs, logits equal the features: the stored class patterns
    # are known, so SHE has well-defined correctly classified training samples
    model = _classifier(num_inputs=10, n_hidden=3, num_outputs=3)
    with torch.no_grad():
        model.layer1.weight.zero_()
        model.layer1.bias.zero_()
        model.layer1.weight[:, :3] = torch.eye(3)
        model.classifier.weight.copy_(torch.eye(3))
        model.classifier.bias.zero_()
    return model


def _she_loader() -> DataLoader:
    x = torch.zeros(24, 10)
    y = torch.arange(24) % 3
    x[torch.arange(24), y] = 4.0
    return DataLoader(TensorDataset(x, y), batch_size=6)


DETECTOR_CASES = [
    # logits
    DetectorCase("MSP", _classifier, MaxSoftmax, VECTOR),
    DetectorCase("TemperatureScaling", _classifier, TemperatureScaling, VECTOR),
    DetectorCase("Entropy", _classifier, Entropy, VECTOR),
    DetectorCase("EnergyBased", _classifier, EnergyBased, VECTOR),
    DetectorCase("MaxLogit", _classifier, MaxLogit, VECTOR),
    DetectorCase("GEN", _classifier, GEN, VECTOR),
    DetectorCase("KLMatching", _classifier, KLMatching, VECTOR),
    DetectorCase("ODIN", _classifier, lambda m: ODIN(m, eps=0.001), VECTOR),
    DetectorCase("OpenMax", _classifier, lambda m: OpenMax(m, tailsize=3, alpha=2), VECTOR),
    DetectorCase("WeightedEBO", _classifier, lambda m: WeightedEBO(m, torch.ones(1, 3)), VECTOR),
    DetectorCase("MCD", _classifier, lambda m: MCD(m, samples=4, mode="var"), VECTOR),
    # features
    DetectorCase("KNN", _classifier, lambda m: KNN(m.features, k=3), VECTOR),
    DetectorCase("GMM", _classifier, lambda m: GMM(m.features), VECTOR),
    DetectorCase(
        "MCM",
        _classifier,
        lambda m: MCM(
            encoder=m.features,
            text_embeddings=F.normalize(
                torch.randn(3, 10, generator=_generator(MODEL_SEED)), dim=-1
            ),
            temperature=1.0,
        ),
        VECTOR,
    ),
    DetectorCase("PNML", _classifier, lambda m: PNML(m.features, m.classifier), VECTOR),
    DetectorCase("NNGuide", _classifier, lambda m: NNGuide(m.features, m.classifier, k=3), VECTOR),
    DetectorCase("fDBD", _classifier, lambda m: fDBD(m.features, m.classifier), VECTOR),
    DetectorCase("Mahalanobis", _classifier, lambda m: Mahalanobis(m.features), VECTOR),
    DetectorCase("RMD", _classifier, lambda m: RMD(m.features), VECTOR),
    DetectorCase(
        "Residual",
        _classifier,
        lambda m: Residual(m.features, d=4, w=m.classifier.weight, b=m.classifier.bias),
        VECTOR,
    ),
    DetectorCase(
        "ViM",
        # D = 16 is a candidate of hyperparameter_space, which covers d == D
        lambda: _classifier(n_hidden=16),
        lambda m: ViM(m.features, d=4, w=m.classifier.weight, b=m.classifier.bias),
        VECTOR,
    ),
    DetectorCase(
        "NCI",
        _classifier,
        lambda m: NCI(encoder=m.features, head=m.classifier, alpha=0.0),
        VECTOR,
    ),
    DetectorCase(
        "SHE",
        _she_classifier,
        lambda m: SHE(m.features, m.classifier),
        VECTOR,
        fit_loader=_she_loader,
    ),
    DetectorCase(
        "DICE",
        _classifier,
        lambda m: DICE(encoder=m.features, w=m.classifier.weight, b=m.classifier.bias, p=0.65),
        VECTOR,
    ),
    DetectorCase("LTS", _classifier, lambda m: LTS(encoder=m.features, head=m.classifier), VECTOR),
    DetectorCase(
        "ReAct",
        _classifier,
        lambda m: ReAct(backbone=m.features, head=m.classifier),
        VECTOR,
    ),
    DetectorCase(
        "VRA", _classifier, lambda m: VRA(backbone=m.features, head=m.classifier), VECTOR
    ),
    DetectorCase(
        "MahalanobisODIN",
        _classifier,
        lambda m: MahalanobisODIN(m.features, eps=0.001),
        VECTOR,
    ),
    # gradients
    DetectorCase(
        "GradUncertainty",
        _gradient_classifier,
        lambda m: GradUncertainty(m, param_filter=lambda n: n.startswith("classifier")),
        VECTOR,
    ),
    DetectorCase(
        "GradNorm",
        _gradient_classifier,
        lambda m: GradNorm(m, param_filter=lambda n: n.startswith("classifier")),
        VECTOR,
    ),
    # feature maps and per-layer inputs
    DetectorCase(
        "ASH",
        _conv,
        lambda m: ASH(backbone=m.feature_maps, head=m.forward_feature_maps),
        IMAGE,
    ),
    DetectorCase(
        "SCALE",
        _conv,
        lambda m: SCALE(backbone=m.feature_maps, head=m.forward_feature_maps),
        IMAGE,
    ),
    DetectorCase(
        "RankFeat",
        _conv,
        lambda m: RankFeat(backbone=m.feature_maps, head=m.forward_feature_maps),
        IMAGE,
    ),
    DetectorCase("MultiMahalanobis", _conv, lambda m: MultiMahalanobis([m.conv1, m.relu]), IMAGE),
    DetectorCase(
        "Gram",
        _conv,
        lambda m: Gram(
            head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), m.fc),
            feature_layers=[m.conv1, m.relu],
            num_classes=3,
            num_poles_list=[1, 2],
        ),
        IMAGE,
    ),
    DetectorCase(
        "NACUE",
        _conv,
        lambda m: NACUE(
            model=m,
            layers=[m.conv1, m.relu],
            m_bins=[10, 10],
            alpha=5.0,
            o_star=50,
        ),
        IMAGE,
    ),
]
