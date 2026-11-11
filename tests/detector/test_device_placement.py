import pytest
import torch
from torch import Tensor, nn

from pytorch_ood.api import Detector, FeatureMapsDetector, FeaturesDetector, LogitsDetector
from tests.helpers import requires
from tests.helpers.detectors import DETECTOR_CASES, DETECTOR_CLASSES, DetectorCase

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


def _producer(case: DetectorCase):
    cls = DETECTOR_CLASSES[case.name]
    return next((p for base, p in PRODUCERS.items() if issubclass(cls, base)), None)


def _tensor_state(name: str, value):
    """Tensors a detector holds, also in containers, modules and nested pytorch-ood objects."""
    if isinstance(value, Tensor):
        yield name, value
    elif isinstance(value, nn.Module):
        yield from ((f"{name}.{k}", t) for k, t in value.named_parameters())
        yield from ((f"{name}.{k}", t) for k, t in value.named_buffers())
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _tensor_state(f"{name}[{i}]", item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_state(f"{name}[{key!r}]", item)
    elif type(value).__module__.startswith("pytorch_ood."):
        for key, item in vars(value).items():
            yield from _tensor_state(f"{name}.{key}", item)


def _assert_on(detector: Detector, scores: Tensor, device: torch.device):
    assert scores.device == device
    elsewhere = [
        f"{name} ({t.device})"
        for key, value in vars(detector).items()
        for name, t in _tensor_state(key, value)
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
    x = torch.randn(8, *case.input_shape, generator=seed).to(accelerator)

    _assert_on(detector, detector(x), accelerator)


@requires(cuda=1)
@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.name) for c in DETECTOR_CASES if _producer(c)]
)
def test_state_stays_on_the_device_of_the_representations(
    case: DetectorCase, seed: torch.Generator, accelerator: torch.device
):
    detector = case.detector(case.model().to(accelerator))
    producer, kind = _producer(case)
    produce = getattr(detector, producer)
    # the producer is removed once it has computed the representations
    with torch.no_grad():
        z = produce(torch.randn(8, *case.input_shape, generator=seed).to(accelerator))
        if detector.requires_fit:
            x, y = (torch.cat(t) for t in zip(*case.make_fit_loader()))
            fit_data = produce(x.to(accelerator)), y.to(accelerator)
    setattr(detector, producer, None)
    if detector.requires_fit:
        getattr(detector, f"fit_{kind}")(*fit_data)

    _assert_on(detector, getattr(detector, f"predict_{kind}")(z), accelerator)
