"""
Properties every detector has, checked for each case in :data:`DETECTOR_CASES` and, where no
instance is needed, for each exported detector class.
"""

import copy
import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union

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
    ModelModeException,
    ModelNotSetException,
    RequiresFittingException,
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
from tests.helpers import requires
from tests.helpers.components import DETECTORS
from tests.helpers.detectors import DETECTOR_CASES, FIT_SEED, DetectorCase
from tests.helpers.snapshot import snapshot


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


# Detector classes per case, to select the cases a property applies to at collection time
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


def _fitted(
    case: DetectorCase,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **loader_kwargs,
) -> Detector:
    detector = case.detector(case.model() if model is None else model)
    if device is not None:
        detector.to(device)
    if detector.requires_fit:
        detector.fit(case.make_fit_loader(**loader_kwargs))
    return detector


def _concatenated(batches: Iterable[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    x, y = zip(*batches)
    return torch.cat(x), torch.cat(y)


#: several negative labels, so that a filter for one particular label does not pass
OOD_LABELS = (-1, -5, -100)


def _ood_labels(n: int, dtype: torch.dtype) -> Tensor:
    return torch.tensor(OOD_LABELS, dtype=dtype)[torch.arange(n) % len(OOD_LABELS)]


def _with_ood_rows(
    loader: DataLoader, generator: torch.Generator, n: int = len(OOD_LABELS)
) -> DataLoader:
    """The same ID samples in the same order, with ``n`` OOD rows in every batch."""
    batches = [
        (
            torch.cat([x, 10 + torch.randn(n, *x.shape[1:], generator=generator)]),
            torch.cat([y, _ood_labels(n, y.dtype)]),
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


@pytest.mark.parametrize("case", _cases())
def test_scores_are_not_differentiable(case: DetectorCase, seed: torch.Generator):
    assert not _fitted(case)(_inputs(case, seed)).requires_grad


@pytest.mark.parametrize("case", _cases(_produced))
def test_scores_on_a_differentiable_representation_are_not_differentiable(
    case: DetectorCase, seed: torch.Generator
):
    detector = _fitted(case)
    representation = _produced(type(detector))
    z = getattr(detector, representation.producer)(_inputs(case, seed))
    assert z.requires_grad  # the model's parameters require grad

    assert not getattr(detector, f"predict_{representation.kind}")(z).requires_grad


def _base_of(cls: type) -> type:
    """The semantic base class of a detector, or :class:`Detector` for one without (MCD)."""
    return next((base for base in SEMANTIC_BASES if issubclass(cls, base)), Detector)


@pytest.mark.parametrize("base", (*SEMANTIC_BASES, Detector), ids=_name)
def test_float32_input_gives_float32_scores(base: type, seed: torch.Generator):
    cases = [c for c in DETECTOR_CASES if _base_of(_CLASSES[c.name]) is base]
    assert cases

    dtypes = {c.name: _fitted(c)(_inputs(c, seed)).dtype for c in cases}

    assert dtypes == {c.name: torch.float32 for c in cases}


def _memory_layouts(x: Tensor) -> Dict[str, Tensor]:
    """The values of ``x``, stored in memory layouts other than the default one."""
    # gaps along every dimension, so that ``view`` cannot merge any two of them; NaN in the gaps,
    # so that code reading the storage directly cannot go unnoticed
    spread = torch.full([2 * size for size in x.shape], float("nan"), dtype=x.dtype)
    every_second = tuple(slice(None, None, 2) for _ in x.shape)
    spread[every_second] = x
    layouts = {"gaps": spread[every_second]}
    if x.ndim == 4:
        layouts["channels_last"] = x.contiguous(memory_format=torch.channels_last)
    return layouts


def _assert_same_scores_in_every_memory_layout(predict: Callable[[Tensor], Tensor], x: Tensor):
    expected = predict(x)
    for name, rearranged in _memory_layouts(x).items():
        # float32 forward passes round differently per layout; a layout bug is far larger or raises
        torch.testing.assert_close(predict(rearranged), expected, rtol=1e-3, atol=1e-3, msg=name)


def _deterministic(cls: type) -> bool:
    # MCD samples its dropout masks from the global RNG, so two calls give different scores
    return cls is not MCD


@pytest.mark.parametrize("case", _cases(_deterministic))
def test_scores_do_not_depend_on_the_memory_layout(case: DetectorCase, seed: torch.Generator):
    _assert_same_scores_in_every_memory_layout(_fitted(case), _inputs(case, seed))


@pytest.mark.parametrize("case", _cases(lambda cls: _produced(cls) and _deterministic(cls)))
def test_scores_on_a_representation_do_not_depend_on_the_memory_layout(
    case: DetectorCase, seed: torch.Generator
):
    detector = _fitted(case)
    representation = _produced(type(detector))
    with torch.no_grad():
        z = getattr(detector, representation.producer)(_inputs(case, seed))

    _assert_same_scores_in_every_memory_layout(
        getattr(detector, f"predict_{representation.kind}"), z
    )


BATCH_DEPENDENT = {
    "OpenMax": pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="OpenMax computes the distance to a class center only for classes that some sample "
        "in the batch is predicted as, and uses 0 for the others among a sample's top alpha",
    ),
}


@pytest.mark.parametrize("case", _cases(_deterministic, marks=BATCH_DEPENDENT))
def test_scores_do_not_depend_on_the_other_samples_in_the_batch(
    case: DetectorCase, seed: torch.Generator
):
    detector = _fitted(case)
    x = _inputs(case, seed)
    expected = detector(x)
    # float32 matrix products round differently for other batch shapes (BLAS picks other kernels
    # and summation orders). Distance-based scores are differences of terms around 20, so they are
    # off by a few ulps of those terms, not of the score: up to 1.5e-5 for RMD over 50 input
    # draws. A score that depends on the batch is off by far more.
    # The tolerance is never wider than a small part of the spread of the scores: ODIN at
    # T=1000 spreads its scores over about 1e-4. Constant scores could not show a dependence.
    spread = (expected.max() - expected.min()).item()
    assert spread > 0
    tolerance = {"rtol": 1e-5, "atol": min(1e-4, 1e-2 * spread)}

    for i in range(len(x)):
        alone = detector(x[i : i + 1])
        torch.testing.assert_close(
            alone, expected[i : i + 1], **tolerance, msg=lambda m: f"sample {i} alone: {m}"
        )
    perm = torch.arange(len(x)).roll(1)  # every sample moves
    torch.testing.assert_close(
        detector(x[perm]), expected[perm], **tolerance, msg=lambda m: f"permuted: {m}"
    )


# --- devices ------------------------------------------------------------------------------------


def _assert_scores_on(scores: Tensor, device: torch.device):
    assert scores.device == device
    assert torch.isfinite(scores).all()


@requires(cuda=1)
@pytest.mark.parametrize("case", _cases())
def test_cpu_inputs_are_scored_on_the_device_of_the_detector(
    case: DetectorCase, seed: torch.Generator, accelerator: torch.device
):
    detector = _fitted(case, device=accelerator)

    assert detector.device == accelerator
    _assert_scores_on(detector(_inputs(case, seed)), accelerator)


@requires(cuda=1)
@pytest.mark.parametrize("case", _cases(_produced))
def test_cpu_representations_are_scored_on_the_device_of_the_detector(
    case: DetectorCase, seed: torch.Generator, accelerator: torch.device
):
    detector = _fitted(case, device=accelerator)
    representation = _produced(type(detector))
    with torch.no_grad():
        z = getattr(detector, representation.producer)(_inputs(case, seed).to(accelerator))

    _assert_scores_on(getattr(detector, f"predict_{representation.kind}")(z.cpu()), accelerator)


#: the arguments of ``predict_structured`` differ per detector
STRUCTURED_INPUTS = {
    "MultiMahalanobis": lambda: ([torch.randn(8, 8), torch.randn(8, 8)],),
    "Gram": lambda: (torch.randn(8, 3), [torch.randn(8, 8, 8, 8), torch.randn(8, 8, 8, 8)]),
}


@requires(cuda=1)
@pytest.mark.parametrize("case", _cases(lambda cls: issubclass(cls, StructuredDetector)))
def test_cpu_structured_inputs_are_scored_on_the_device_of_the_detector(
    case: DetectorCase, accelerator: torch.device
):
    detector = _fitted(case, device=accelerator)

    _assert_scores_on(detector.predict_structured(*STRUCTURED_INPUTS[case.name]()), accelerator)


# --- preconditions ------------------------------------------------------------------------------


def _producer_name(cls: type) -> str:
    """The constructor parameter that holds the model, encoder, backbone, layers or head."""
    return list(inspect.signature(cls.__init__).parameters)[1]


def _without_producer(detector: Detector) -> Detector:
    setattr(detector, _producer_name(type(detector)), None)
    return detector


@pytest.mark.parametrize("case", _cases(_fits))
def test_predict_before_fit_raises(case: DetectorCase, seed: torch.Generator):
    with pytest.raises(RequiresFittingException):
        case.build()(_inputs(case, seed))


@pytest.mark.parametrize("case", _cases(lambda cls: _produced(cls) and _fits(cls)))
def test_predict_on_representation_before_fit_raises(case: DetectorCase, seed: torch.Generator):
    detector = case.build()
    representation = _produced(type(detector))
    with torch.no_grad():
        z = getattr(detector, representation.producer)(_inputs(case, seed))

    with pytest.raises(RequiresFittingException):
        getattr(detector, f"predict_{representation.kind}")(z)


@pytest.mark.parametrize("case", _cases())
def test_predict_without_producer_raises(case: DetectorCase, seed: torch.Generator):
    detector = _without_producer(_fitted(case))

    with pytest.raises(ModelNotSetException):
        detector(_inputs(case, seed))


@pytest.mark.parametrize("case", _cases(_fits))
def test_fit_without_producer_raises(case: DetectorCase):
    with pytest.raises(ModelNotSetException):
        _without_producer(case.build()).fit(case.make_fit_loader())


# --- fitting ------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases(lambda cls: _produced(cls) and _fits(cls)))
def test_fit_equals_fit_on_representation(case: DetectorCase, seed: torch.Generator):
    detector = case.build()
    twin = copy.deepcopy(detector)
    representation = _produced(type(detector))
    loader = case.make_fit_loader()

    detector.fit(loader)
    # batch by batch, like fit(): one large batch rounds differently in float32, which
    # ill-conditioned covariances (GMM, RMD) amplify beyond any sensible tolerance
    producer = getattr(twin, representation.producer)
    with torch.no_grad():
        z, y = _concatenated([(producer(xb), yb) for xb, yb in loader])
    getattr(twin, f"fit_{representation.kind}")(z, y)

    x = _inputs(case, seed)
    torch.testing.assert_close(detector(x), twin(x))


#: enough samples that Gram holds out a validation split (10 % per class)
REFIT_SAMPLES = 60


@pytest.mark.parametrize("case", _cases(_fits))
def test_fitting_twice_gives_the_same_scores(case: DetectorCase, seed: torch.Generator):
    detector = case.build()
    loader = case.make_fit_loader(n_samples=REFIT_SAMPLES)
    x = _inputs(case, seed)

    once = detector.fit(loader)(x)

    torch.testing.assert_close(detector.fit(loader)(x), once)


@pytest.mark.parametrize("case", _cases(_fits))
def test_refit_equals_a_fresh_fit(case: DetectorCase, seed: torch.Generator):
    # GridSearch refits one instance per candidate: nothing from an earlier fit may survive
    detector = case.build()
    fresh = copy.deepcopy(detector)
    earlier = case.make_fit_loader(n_samples=REFIT_SAMPLES, seed=FIT_SEED + 1)  # different data
    loader = case.make_fit_loader(n_samples=REFIT_SAMPLES)
    x = _inputs(case, seed)

    detector.fit(earlier).fit(loader)

    torch.testing.assert_close(detector(x), fresh.fit(loader)(x))


@contextmanager
def _no_ood_warning() -> Iterator[None]:
    """Fails on a warning about OOD samples, e.g. for fit data with ID samples only."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*OOD", category=UserWarning)
        yield


@pytest.mark.parametrize("case", _cases(_fits))
def test_fit_ignores_ood_rows_in_the_loader(case: DetectorCase, seed: torch.Generator):
    detector = case.build()
    twin = copy.deepcopy(detector)
    loader = case.make_fit_loader()

    with _no_ood_warning():
        detector.fit(loader)
    with pytest.warns(UserWarning, match="OOD"):
        twin.fit(_with_ood_rows(loader, seed))

    x = _inputs(case, seed)
    torch.testing.assert_close(detector(x), twin(x))


#: the per-layer inputs of ``fit_structured`` for raw inputs, for structured detectors that have
#: it (Gram fits only from a loader)
STRUCTURED_REPRESENTATIONS: Dict[type, Callable[[Detector, Tensor], List[Tensor]]] = {
    MultiMahalanobis: lambda detector, x: [
        nn.Sequential(*detector.model[: i + 1])(x).mean(dim=(2, 3))
        for i in range(len(detector.model))
    ],
}


def _fits_on_representation(cls: type) -> bool:
    return _fits(cls) and (_produced(cls) is not None or cls in STRUCTURED_REPRESENTATIONS)


Representations = Union[Tensor, List[Tensor]]


def _representations(detector: Detector, loader: DataLoader) -> Tuple[Representations, Tensor]:
    """The arguments of ``fit_<kind>``, for all samples of ``loader``."""
    produced = _produced(type(detector))
    if produced:
        producer = getattr(detector, produced.producer)
        with torch.no_grad():
            return _concatenated([(producer(x), y) for x, y in loader])
    represent = STRUCTURED_REPRESENTATIONS[type(detector)]
    with torch.no_grad():
        zs, y = zip(*[(represent(detector, x), y) for x, y in loader])
    return [torch.cat(layer) for layer in zip(*zs)], torch.cat(y)


def _rows(z: Representations, index: Tensor) -> Representations:
    return [t[index] for t in z] if isinstance(z, list) else z[index]


def _fit_on_representation(detector: Detector) -> Callable[..., Detector]:
    return getattr(detector, f"fit_{_representation(type(detector)).kind}")


@pytest.mark.parametrize("case", _cases(_fits_on_representation))
def test_fit_on_representation_drops_ood_rows_with_a_warning(
    case: DetectorCase, seed: torch.Generator
):
    detector = case.build()
    twin = copy.deepcopy(detector)
    z, y = _representations(detector, _with_ood_rows(case.make_fit_loader(), seed))
    known = y >= 0

    with _no_ood_warning():
        _fit_on_representation(detector)(_rows(z, known), y[known])
    with pytest.warns(UserWarning, match="OOD"):
        _fit_on_representation(twin)(z, y)

    x = _inputs(case, seed)
    torch.testing.assert_close(detector(x), twin(x))


def _only_ood(loader: DataLoader) -> DataLoader:
    x, y = _concatenated(loader)
    return DataLoader(TensorDataset(x, _ood_labels(len(y), y.dtype)), batch_size=loader.batch_size)


#: the message of the ValueError for fit data without ID samples (``utils.utils._NO_ID_SAMPLES``)
NO_ID_SAMPLES = "No ID samples to fit on"


@pytest.mark.filterwarnings("ignore:.*OOD")
@pytest.mark.parametrize("case", _cases(_fits))
def test_fit_without_id_samples_raises(case: DetectorCase):
    with pytest.raises(ValueError, match=NO_ID_SAMPLES):
        case.build().fit(_only_ood(case.make_fit_loader()))


@pytest.mark.filterwarnings("ignore:.*OOD")
@pytest.mark.parametrize("case", _cases(_fits_on_representation))
def test_fit_on_representation_without_id_samples_raises(case: DetectorCase):
    detector = case.build()
    z, y = _representations(detector, case.make_fit_loader())

    with pytest.raises(ValueError, match=NO_ID_SAMPLES):
        _fit_on_representation(detector)(z, _ood_labels(len(y), y.dtype))


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


def _state(model: nn.Module) -> Dict[str, object]:
    """Everything reachable from the model, and torch's global module hooks."""
    global_hooks = {
        f"torch.{name}": registry
        for name, registry in vars(nn.modules.module).items()
        if name.startswith("_global") and isinstance(registry, dict)
    }
    return snapshot(model=model, **global_hooks)


@pytest.mark.parametrize("gradients", [None, 7.0], ids=["no_gradients", "existing_gradients"])
@pytest.mark.parametrize("case", _cases())
def test_model_is_unchanged(case: DetectorCase, gradients: Optional[float], seed: torch.Generator):
    model = case.model()
    for p in model.parameters():
        p.grad = None if gradients is None else torch.full_like(p, gradients)
    before = _state(model)

    detector = _fitted(case, model)
    assert _state(model) == before  # checked here too: predict could undo what fit left
    detector(_inputs(case, seed))
    assert _state(model) == before


def _in_mode(model: nn.Module, mode: str) -> nn.Module:
    """The ``mixed`` modes differ in the root, on which the mode handling of a detector may branch."""
    model.train(mode in ("train", "mixed_under_train"))
    if mode.startswith("mixed"):
        for i, child in enumerate(model.children()):
            child.train(i % 2 == 1)
    return model


#: cannot score a model with modules in training mode and raise ModelModeException instead
#: (docs/info.rst, design choice 4)
_REFUSE_TRAINING_MODE = (GradNorm, GradUncertainty)


def _modes(model: nn.Module) -> Dict[str, bool]:
    return {n: m.training for n, m in model.named_modules()}


@pytest.mark.parametrize("mode", ["eval", "train", "mixed_under_train", "mixed_under_eval"])
@pytest.mark.parametrize("case", _cases())
def test_model_mode_is_unchanged(case: DetectorCase, mode: str, seed: torch.Generator):
    model = _in_mode(case.model(), mode)
    before = _modes(model)
    detector = _fitted(case, model)

    if mode != "eval" and _CLASSES[case.name] in _REFUSE_TRAINING_MODE:
        with pytest.raises(ModelModeException, match=r"model\.eval\(\)"):
            detector(_inputs(case, seed))
    else:
        detector(_inputs(case, seed))

    assert _modes(model) == before


@pytest.mark.parametrize("case", _cases())
def test_model_is_restored_when_the_forward_pass_fails(case: DetectorCase, seed: torch.Generator):
    # mixed, so that a detector that switches modes has to restore each submodule
    mode = "eval" if _CLASSES[case.name] in _REFUSE_TRAINING_MODE else "mixed_under_eval"
    model = _in_mode(case.model(), mode)
    before = _state(model)
    detector = _fitted(case, model)

    def fail(*_):
        raise RuntimeError("forward failed")

    # on every module, since a detector may call only a part of the model (e.g. its encoder)
    handles = [module.register_forward_hook(fail) for module in model.modules()]
    with pytest.raises(RuntimeError, match="forward failed"):
        detector(_inputs(case, seed))
    for handle in handles:
        handle.remove()

    assert _state(model) == before


@pytest.mark.parametrize("case", _cases())
def test_frozen_model_gives_the_same_scores(case: DetectorCase, seed: torch.Generator):
    model = case.model()
    frozen = copy.deepcopy(model).requires_grad_(False)
    x = _inputs(case, seed)

    with torch.random.fork_rng():
        torch.manual_seed(0)  # MCD samples dropout masks from the global RNG
        expected = _fitted(case, model)(x)
        torch.manual_seed(0)
        scores = _fitted(case, frozen)(x)

    assert not any(p.requires_grad for p in frozen.parameters())
    torch.testing.assert_close(scores, expected)


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
    assert Detector._fit_uses_ood is False
    assert not any(hasattr(Detector, f"{p}_{kind}") for kind in KINDS for p in ("predict", "fit"))


def test_fit_method_without_labels_is_rejected_at_class_creation():
    # the fit_<kind> wrapper finds the labels as the second parameter
    with pytest.raises(TypeError, match="labels"):

        class NoLabels(FeaturesDetector):
            def fit_features(self, z: Tensor) -> "NoLabels":
                return self


def test_predict_structured_is_not_differentiable():
    # the expected arguments differ per detector, so the contract cases cannot call it generically
    class Summing(StructuredDetector):
        def predict_structured(self, z: Tensor) -> Tensor:
            return z.sum(dim=1)

        def predict(self, x: Tensor) -> Tensor:
            return self.predict_structured(x)

        def fit(self, data_loader: DataLoader) -> "Summing":
            return self

    assert not Summing().predict_structured(torch.ones(2, 3, requires_grad=True)).requires_grad


def test_fit_structured_signature_is_up_to_the_detector():
    # its arguments depend on the detector, so the second one need not be the labels
    class Mixed(StructuredDetector):
        def fit_structured(self, logits: Tensor, features: Tensor, y: Tensor) -> "Mixed":
            self.fitted_on = logits, features, y
            return self

        def predict_structured(self, logits: Tensor, features: Tensor) -> Tensor:
            return logits.sum(dim=1)

        def predict(self, x: Tensor) -> Tensor:
            return x.sum(dim=1)

        def fit(self, data_loader: DataLoader) -> "Mixed":
            return self

    logits, features, y = (
        torch.zeros(3, 2),
        torch.tensor([-1.0, -2.0, 1.0]),
        torch.tensor([0, 1, 2]),
    )

    fitted_on = Mixed().fit_structured(logits, features, y).fitted_on

    assert [len(t) for t in fitted_on] == [3, 3, 3]


@pytest.mark.filterwarnings("error")
def test_fit_uses_ood_passes_ood_rows_through():
    class UsesOOD(FeaturesDetector):
        _fit_uses_ood = True

        def fit_features(self, z: Tensor, y: Tensor) -> "UsesOOD":
            self.fitted_on = y
            return self

        def predict_features(self, z: Tensor) -> Tensor:
            return z.sum(dim=1)

    y = torch.tensor([0, -1, 1, -1])

    assert torch.equal(UsesOOD().fit_features(torch.zeros(4, 2), y).fitted_on, y)


# --- default hyperparameters ---------------------------------------------------------------------

#: defaults that decide the published results of a method
DEFAULTS = [
    (fDBD, "distance_as_normalizer", True),
    (GradNorm, "temperature", 1.0),
    (KNN, "k", 50),
    (KNN, "normalize", True),
    (LTS, "p", 0.05),
    (Mahalanobis, "eps", 1e-10),
    (Mahalanobis, "use_pinv", False),
    (ODIN, "eps", 0.0014),
    (ODIN, "temperature", 1000.0),
]


@pytest.mark.parametrize("cls, name, default", DEFAULTS)
def test_default_hyperparameter(cls: type, name: str, default):
    assert inspect.signature(cls.__init__).parameters[name].default == default


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
