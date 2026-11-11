from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor, nn

from pytorch_ood.api import Detector, FeatureMapsDetector, FeaturesDetector, LogitsDetector
from tests.helpers import requires
from tests.helpers.detectors import DETECTOR_CASES, DetectorCase

#: producer attribute and representation kind of each base class with a ``predict_<kind>``
PRODUCERS = {
    LogitsDetector: ("model", "logits"),
    FeaturesDetector: ("encoder", "features"),
    FeatureMapsDetector: ("backbone", "feature_maps"),
}

XFAIL = {
    "NACUE": pytest.mark.xfail(
        reason='NACUE creates its histogram counts with device="cpu"; it is reworked separately',
        strict=True,
    )
}


def _producer(case: DetectorCase) -> Optional[Tuple[str, str]]:
    cls = type(case.build())
    return next((p for base, p in PRODUCERS.items() if issubclass(cls, base)), None)


def _inputs(case: DetectorCase, generator: torch.Generator, n: int = 8) -> Tensor:
    return torch.randn(n, *case.input_shape, generator=generator)


def _tensor_state(name: str, value, seen: set):
    """Tensors a detector holds, also in containers, modules and nested pytorch-ood objects."""
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Tensor):
        yield name, value
    elif isinstance(value, nn.Module):
        yield from ((f"{name}.{k}", t) for k, t in value.named_parameters())
        yield from ((f"{name}.{k}", t) for k, t in value.named_buffers())
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _tensor_state(f"{name}[{i}]", item, seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_state(f"{name}[{key!r}]", item, seen)
    elif type(value).__module__.startswith("pytorch_ood."):
        for key, item in vars(value).items():
            yield from _tensor_state(f"{name}.{key}", item, seen)


def _assert_on(detector: Detector, scores: Tensor, device: torch.device):
    assert scores.device == device
    seen: set = set()
    elsewhere = [
        f"{name} ({t.device})"
        for key, value in vars(detector).items()
        for name, t in _tensor_state(key, value, seen)
        if t.device != device
    ]
    assert elsewhere == [], f"state not on {device}"


@requires(cuda=1)
@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.name, marks=XFAIL.get(c.name, ())) for c in DETECTOR_CASES]
)
def test_state_stays_on_the_device_of_the_model(
    case: DetectorCase, seed: torch.Generator, accelerator: torch.device
):
    detector = case.detector(case.model().to(accelerator))
    if detector.requires_fit:
        detector.fit(case.make_fit_loader())

    _assert_on(detector, detector(_inputs(case, seed).to(accelerator)), accelerator)


@requires(cuda=1)
@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.name) for c in DETECTOR_CASES if _producer(c)]
)
def test_state_stays_on_the_device_of_the_representations(
    case: DetectorCase, seed: torch.Generator, accelerator: torch.device
):
    # the detector is built without the producer, as for scoring cached representations
    detector = case.detector(case.model().to(accelerator))
    producer, kind = _producer(case)
    produce = getattr(detector, producer)
    with torch.no_grad():
        fit_data = [
            (produce(x.to(accelerator)), y.to(accelerator)) for x, y in case.make_fit_loader()
        ]
        z = produce(_inputs(case, seed).to(accelerator))
    setattr(detector, producer, None)
    if detector.requires_fit:
        zs, ys = zip(*fit_data)
        getattr(detector, f"fit_{kind}")(torch.cat(zs), torch.cat(ys))

    _assert_on(detector, getattr(detector, f"predict_{kind}")(z), accelerator)
