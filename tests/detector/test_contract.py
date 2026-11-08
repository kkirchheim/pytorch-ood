"""
Properties every detector has, checked for each case in :data:`DETECTOR_CASES` and, where no
instance is needed, for each exported detector class.
"""

import copy
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import (
    Detector,
    FeatureMapsDetector,
    FeaturesDetector,
    GradientDetector,
    LogitsDetector,
    StructuredDetector,
)
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
from tests.helpers.components import DETECTORS
from tests.helpers.detectors import DETECTOR_CASES, DetectorCase


@dataclass(frozen=True)
class Representation:
    """The second entry point a semantic base class adds next to ``predict(x)`` and ``fit``."""

    #: name of the constructor parameter and attribute that produces the representation from raw
    #: inputs; ``None`` if there is no single producer
    producer: Optional[str]
    #: suffix of ``predict_*`` and ``fit_*``
    kind: str


REPRESENTATIONS: Dict[type, Representation] = {
    LogitsDetector: Representation("model", "logits"),
    FeaturesDetector: Representation("encoder", "features"),
    FeatureMapsDetector: Representation("backbone", "feature_maps"),
    StructuredDetector: Representation(None, "structured"),
}
SEMANTIC_BASES = (*REPRESENTATIONS, GradientDetector)


def _representation(cls: type) -> Optional[Representation]:
    return next((r for base, r in REPRESENTATIONS.items() if issubclass(cls, base)), None)


def _produced(cls: type) -> Optional[Representation]:
    """The representation, if the detector computes it from raw inputs with a producer."""
    representation = _representation(cls)
    return representation if representation and representation.producer else None


def _name(cls: type) -> str:
    return cls.__name__


# Detector classes per case, to select the cases a property applies to at collection time. The
# builders draw random weights, which must not shift the global RNG for the tests that follow.
with torch.random.fork_rng():
    _CLASSES: Dict[str, Type[Detector]] = {c.name: type(c.build()) for c in DETECTOR_CASES}


def _cases(applies=lambda cls: True, marks: Optional[Dict[str, pytest.MarkDecorator]] = None):
    marks = marks or {}
    return [
        pytest.param(c, id=c.name, marks=marks.get(c.name, ()))
        for c in DETECTOR_CASES
        if applies(_CLASSES[c.name])
    ]


def _fits(cls: type) -> bool:
    return cls.requires_fit


def _inputs(case: DetectorCase, generator: torch.Generator, n: int = 8) -> Tensor:
    return torch.randn(n, *case.input_shape, generator=generator)


def _fitted(case: DetectorCase, model: Optional[nn.Module] = None, **loader_kwargs) -> Detector:
    detector = case.detector(case.model() if model is None else model)
    if detector.requires_fit:
        detector.fit(case.make_fit_loader(**loader_kwargs))
    return detector


def _concatenated(loader: DataLoader) -> Tuple[Tensor, Tensor]:
    x, y = zip(*loader)
    return torch.cat(x), torch.cat(y)


def _with_ood_rows(loader: DataLoader, generator: torch.Generator, n: int = 2) -> DataLoader:
    """The same ID samples in the same order, with ``n`` OOD rows (label -1) in every batch."""
    batches = [
        (
            torch.cat([x, 10 + torch.randn(n, *x.shape[1:], generator=generator)]),
            torch.cat([y, torch.full((n,), -1, dtype=y.dtype)]),
        )
        for x, y in loader
    ]
    x, y = _concatenated(batches)
    return DataLoader(TensorDataset(x, y), batch_size=loader.batch_size + n)


# --- scores -------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases())
def test_one_finite_score_per_input(case: DetectorCase, seed: torch.Generator):
    detector = _fitted(case)
    x = _inputs(case, seed)

    scores = detector(x)

    assert isinstance(scores, Tensor)
    assert scores.shape == (len(x),)
    assert torch.isfinite(scores).all()
    assert detector(x[:1]).shape == (1,)


@pytest.mark.parametrize("case", _cases(_produced))
def test_predict_equals_predict_on_representation(case: DetectorCase, seed: torch.Generator):
    detector = _fitted(case)
    representation = _produced(type(detector))
    x = _inputs(case, seed)

    with torch.no_grad():
        z = getattr(detector, representation.producer)(x)
    expected = getattr(detector, f"predict_{representation.kind}")(z)

    torch.testing.assert_close(detector(x), expected)


# --- fitting ------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases(lambda cls: _produced(cls) and _fits(cls)))
def test_fit_equals_fit_on_representation(case: DetectorCase, seed: torch.Generator):
    detector = case.build()
    twin = copy.deepcopy(detector)
    representation = _produced(type(detector))
    loader = case.make_fit_loader()

    detector.fit(loader)
    x, y = _concatenated(loader)
    with torch.no_grad():
        z = getattr(twin, representation.producer)(x)
    getattr(twin, f"fit_{representation.kind}")(z, y)

    x = _inputs(case, seed)
    torch.testing.assert_close(detector(x), twin(x))


@pytest.mark.parametrize("case", _cases(_fits))
def test_fit_ignores_ood_rows_in_the_loader(case: DetectorCase, seed: torch.Generator):
    detector = case.build()
    twin = copy.deepcopy(detector)
    loader = case.make_fit_loader()

    detector.fit(loader)
    twin.fit(_with_ood_rows(loader, seed))

    x = _inputs(case, seed)
    torch.testing.assert_close(detector(x), twin(x))


@pytest.mark.parametrize(
    "case, name, value",
    [
        pytest.param(c, name, value, id=f"{c.name}-{name}={value}")
        for c in DETECTOR_CASES
        for name, values in _CLASSES[c.name].hyperparameter_space.items()
        for value in values
    ],
)
def test_every_hyperparameter_candidate_gives_finite_scores(
    case: DetectorCase, name: str, value, seed: torch.Generator
):
    detector = case.build().set_hyperparameters(**{name: value})
    if detector.requires_fit:
        # large enough for every candidate, e.g. KNN's k up to 1000
        detector.fit(case.make_fit_loader(n_samples=1024, batch_size=256))

    assert torch.isfinite(detector(_inputs(case, seed))).all()


# --- side effects on the model ------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases())
def test_model_parameters_and_buffers_are_unchanged(case: DetectorCase, seed: torch.Generator):
    model = case.model()
    before = copy.deepcopy(model.state_dict())

    _fitted(case, model)(_inputs(case, seed))

    after = model.state_dict()
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert torch.equal(after[key], value), key


@pytest.mark.parametrize(
    "case",
    _cases(
        marks={
            "MahalanobisODIN": pytest.mark.xfail(
                strict=True, reason="leaves .grad on the encoder's parameters"
            )
        }
    ),
)
def test_no_gradients_left_on_the_model(case: DetectorCase, seed: torch.Generator):
    model = case.model()

    _fitted(case, model)(_inputs(case, seed))

    assert [n for n, p in model.named_parameters() if p.grad is not None] == []


@pytest.mark.parametrize("case", _cases())
def test_model_stays_in_eval_mode(case: DetectorCase, seed: torch.Generator):
    model = case.model()

    _fitted(case, model)(_inputs(case, seed))

    assert [n for n, m in model.named_modules() if m.training] == []


# --- taxonomy -----------------------------------------------------------------------------------

#: the semantic base class (``Detector`` for none) and ``requires_fit`` of every exported detector
TAXONOMY: Dict[type, Tuple[type, bool]] = {
    # logits
    EnergyBased: (LogitsDetector, False),
    Entropy: (LogitsDetector, False),
    GEN: (LogitsDetector, False),
    KLMatching: (LogitsDetector, True),
    MaxLogit: (LogitsDetector, False),
    MaxSoftmax: (LogitsDetector, False),
    OpenMax: (LogitsDetector, True),
    TemperatureScaling: (LogitsDetector, True),
    WeightedEBO: (LogitsDetector, False),
    # features
    DICE: (FeaturesDetector, True),
    fDBD: (FeaturesDetector, True),
    GMM: (FeaturesDetector, True),
    KNN: (FeaturesDetector, True),
    LTS: (FeaturesDetector, False),
    Mahalanobis: (FeaturesDetector, True),
    MCM: (FeaturesDetector, False),
    NCI: (FeaturesDetector, True),
    NNGuide: (FeaturesDetector, True),
    PNML: (FeaturesDetector, True),
    Residual: (FeaturesDetector, True),
    RMD: (FeaturesDetector, True),
    SHE: (FeaturesDetector, True),
    ViM: (FeaturesDetector, True),
    # feature maps
    ASH: (FeatureMapsDetector, False),
    RankFeat: (FeatureMapsDetector, False),
    ReAct: (FeatureMapsDetector, True),
    SCALE: (FeatureMapsDetector, False),
    VRA: (FeatureMapsDetector, True),
    # structured
    Gram: (StructuredDetector, True),
    MultiMahalanobis: (StructuredDetector, True),
    # gradients
    GradNorm: (GradientDetector, False),
    GradUncertainty: (GradientDetector, False),
    MahalanobisODIN: (GradientDetector, True),
    NACUE: (GradientDetector, True),
    ODIN: (GradientDetector, False),
    # none
    MCD: (Detector, False),
}

KINDS = [r.kind for r in REPRESENTATIONS.values()]
PREDICT_METHODS = {f"predict_{kind}" for kind in KINDS}


def test_taxonomy_lists_every_exported_detector():
    assert set(TAXONOMY) == set(DETECTORS)


@pytest.mark.parametrize("cls", TAXONOMY, ids=_name)
def test_semantic_base_and_fit_requirement(cls: type):
    base, requires_fit = TAXONOMY[cls]

    assert [b for b in SEMANTIC_BASES if issubclass(cls, b)] == (
        [] if base is Detector else [base]
    )
    assert cls.requires_fit is requires_fit


@pytest.mark.parametrize("cls", TAXONOMY, ids=_name)
def test_predicts_only_on_the_own_representation(cls: type):
    representation = _representation(cls)
    kind = representation.kind if representation else None

    assert {m for m in PREDICT_METHODS if hasattr(cls, m)} == (
        {f"predict_{kind}"} if kind else set()
    )
    if kind:
        assert hasattr(cls, f"fit_{kind}")


@pytest.mark.parametrize("cls", [c for c in TAXONOMY if _produced(c)], ids=_name)
def test_model_parameter_is_named_after_the_representation(cls: type):
    # benchmark/base.py picks the cached representation to feed a detector by this name
    parameters = list(inspect.signature(cls.__init__).parameters)

    assert parameters[1] == _produced(cls).producer


@pytest.mark.parametrize(
    "cls, method",
    [
        pytest.param(cls, m, id=f"{cls.__name__}.{m}")
        for cls in TAXONOMY
        for m in sorted(PREDICT_METHODS)
        if m in cls.__dict__
    ],
)
def test_predict_methods_keep_their_signature(cls: type, method: str):
    # the wrappers that move tensor arguments to the detector's device must keep the signature of
    # the wrapped method, which the docs show
    parameters = list(inspect.signature(getattr(cls, method)).parameters.values())

    assert parameters[0].name == "self"
    assert inspect.Parameter.VAR_POSITIONAL not in {p.kind for p in parameters}


def test_detector_base_defaults():
    assert Detector.requires_fit is False
    assert not any(hasattr(Detector, f"{p}_{kind}") for kind in KINDS for p in ("predict", "fit"))


# --- fraction hyperparameters -------------------------------------------------------------------


def _identity(x):
    return x


#: hyperparameters that are fractions in [0, 1], checked with ``_check_fraction``
FRACTIONS = {
    "ASH.percentile": lambda v: ASH(_identity, _identity, percentile=v),
    "SCALE.percentile": lambda v: SCALE(_identity, _identity, percentile=v),
    "ReAct.percentile": lambda v: ReAct(_identity, _identity, percentile=v),
    "LTS.p": lambda v: LTS(_identity, torch.nn.Identity(), p=v),
    "DICE.p": lambda v: DICE(_identity, torch.zeros(3, 4), torch.zeros(3), p=v),
    "VRA.lower_percentile": lambda v: VRA(_identity, _identity, lower_percentile=v),
    "VRA.upper_percentile": lambda v: VRA(_identity, _identity, upper_percentile=v),
}


@pytest.mark.parametrize("name", FRACTIONS)
def test_fraction_accepts_values_in_unit_interval(name: str):
    FRACTIONS[name](0.99 if "upper" in name else 0.5)


@pytest.mark.parametrize("name", FRACTIONS)
def test_fraction_rejects_percent_with_hint(name: str):
    with pytest.raises(ValueError, match=r"use 0\.95 instead of 95"):
        FRACTIONS[name](95.0)


@pytest.mark.parametrize("name", FRACTIONS)
def test_fraction_rejects_negative(name: str):
    with pytest.raises(ValueError):
        FRACTIONS[name](-0.1)


def test_vra_rejects_lower_percentile_above_upper():
    with pytest.raises(ValueError):
        VRA(_identity, _identity, lower_percentile=0.9, upper_percentile=0.5)
