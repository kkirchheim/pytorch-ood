import shutil
import tempfile
import warnings

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import FeaturesDetector
from pytorch_ood.benchmark import Benchmark
from pytorch_ood.benchmark import base as base_module
from pytorch_ood.detector import EnergyBased, MahalanobisODIN, MaxSoftmax


class ToyBenchmark(Benchmark):
    def __init__(self):
        self._train = TensorDataset(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                ]
            ),
            torch.tensor([0, 1, 0, 0]),
        )
        self._test_id = TensorDataset(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            torch.tensor([0, 1]),
        )
        self._test_ood = TensorDataset(
            torch.tensor(
                [
                    [3.0, 3.0],
                    [-2.0, 4.0],
                ]
            ),
            torch.tensor([-1, -1]),
        )
        self.ood_names = ["ToyOOD"]

    def train_set(self):
        return self._train

    def test_sets(self, known=True, unknown=True):
        if known and unknown:
            return [self._test_id + self._test_ood]

        if known and not unknown:
            return [self._test_id]

        if not known and unknown:
            return [self._test_ood]

        raise ValueError


class CountingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 1.0]]))

    def forward(self, x):
        self.calls += 1
        return self.linear(x.float())


class CountingFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("unused", torch.zeros(1))  # a cache needs a tensor to hash
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.float()

    def feature_maps(self, x):
        self.calls += 1
        return (2 * x).float()


class SummedFeaturesDetector(FeaturesDetector):
    def __init__(self, model):
        self.encoder = model

    def predict(self, x):
        return self.predict_features(self.encoder(x))

    def predict_features(self, x):
        return -x.sum(dim=1)


def test_benchmark_evaluate_single_detector_keeps_output_shape():
    benchmark = ToyBenchmark()
    detector = MaxSoftmax(CountingLinear())

    results = benchmark.evaluate(detector, loader_kwargs={"batch_size": 2})

    assert len(results) == 1
    assert results[0]["Dataset"] == "ToyOOD"
    assert "Detector" not in results[0]


def test_benchmark_evaluate_list_shares_logits_within_call():
    benchmark = ToyBenchmark()
    model = CountingLinear()

    results = benchmark.evaluate(
        [MaxSoftmax(model), EnergyBased(model)],
        loader_kwargs={"batch_size": 2},
    )

    assert len(results) == 2
    assert {result["Detector"] for result in results} == {"MaxSoftmax", "EnergyBased"}
    assert model.calls == 2


def test_benchmark_reuses_disk_cache_across_benchmark_instances():
    benchmark = ToyBenchmark()
    model = CountingLinear()

    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            benchmark.evaluate(
                MaxSoftmax(model),
                loader_kwargs={"batch_size": 2},
                cache=True,
                cache_dir=tmpdir,
                cache_key="toy-model-v1",
            )

        assert model.calls == 2
        assert len(caught) == 1
        assert "cache_key" in str(caught[0].message)

        second_benchmark = ToyBenchmark()
        second_model = CountingLinear()

        second_benchmark.evaluate(
            EnergyBased(second_model),
            loader_kwargs={"batch_size": 2},
            cache=True,
            cache_dir=tmpdir,
            cache_key="toy-model-v1",
        )

        assert second_model.calls == 0


def test_benchmark_shares_feature_cache_per_method_across_calls():
    benchmark = ToyBenchmark()
    model = CountingFeatures()

    # a bound method is a new object on every access, but the same producer
    detectors = [
        SummedFeaturesDetector(model.forward),
        SummedFeaturesDetector(model.forward),
        SummedFeaturesDetector(model.feature_maps),
    ]
    loader = {"loader_kwargs": {"batch_size": 2}}

    benchmark.evaluate(detectors, **loader)
    assert model.calls == 4  # 2 batches for each of the 2 methods

    benchmark.evaluate(detectors, **loader, cache=True)
    assert model.calls == 8

    benchmark.evaluate(detectors, **loader, cache=True)
    assert model.calls == 8


def test_benchmark_falls_back_for_mahalanobis_with_input_preprocessing():
    benchmark = ToyBenchmark()
    model = CountingFeatures()
    detector = MahalanobisODIN(encoder=model, eps=0.1)
    detector.fit(DataLoader(benchmark.train_set(), batch_size=2))

    calls_after_fit = model.calls

    benchmark.evaluate(
        detector,
        loader_kwargs={"batch_size": 2},
        cache=True,
    )
    calls_after_first_eval = model.calls

    benchmark.evaluate(
        detector,
        loader_kwargs={"batch_size": 2},
        cache=True,
    )

    assert calls_after_first_eval > calls_after_fit
    assert model.calls > calls_after_first_eval


LOADER = {"loader_kwargs": {"batch_size": 2}}
WEIGHTS_A = [[1.0, 0.5], [-0.25, 1.0]]
WEIGHTS_B = [[1.0, 0.0], [0.0, 1.0]]


class RenamedBenchmark(ToyBenchmark):
    """The same data as ``ToyBenchmark``, under another name."""


class TwoSetBenchmark(ToyBenchmark):
    def __init__(self):
        super().__init__()
        self.ood_names = ["ToyOOD", "FarOOD"]
        self._far = TensorDataset(torch.tensor([[0.1, 0.1], [0.2, 0.1]]), torch.tensor([-1, -1]))

    def test_sets(self, known=True, unknown=True):
        return super().test_sets(known, unknown) + [self._test_id + self._far]


class BufferLinear(nn.Module):
    """A linear map whose weights are a buffer."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, x):
        return x.float() @ self.weights.T


class ParameterFree(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _linear_with(weights) -> CountingLinear:
    model = CountingLinear()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(weights))
    return model


@pytest.mark.parametrize("make_model", [_linear_with, BufferLinear], ids=["weights", "buffers"])
@pytest.mark.parametrize(
    "cache, on_disk", [(True, False), (False, True), (True, True)], ids=["memory", "disk", "both"]
)
def test_cached_results_match_uncached_for_models_of_one_class(
    make_model, cache, on_disk, tmp_path
):
    def auroc(benchmark, models, **kwargs):
        results = benchmark.evaluate([MaxSoftmax(m) for m in models], **LOADER, **kwargs)
        return [r["AUROC"] for i in range(len(models)) for r in results[i :: len(models)]]

    def cached(name):
        return {
            "cache": cache,
            "cache_key": "toy",
            **({"cache_dir": str(tmp_path / name)} if on_disk else {}),
        }

    a, b = make_model(WEIGHTS_A), make_model(WEIGHTS_B)
    expected = auroc(TwoSetBenchmark(), [a, b])
    assert expected[:2] != expected[2:] and expected[0] != expected[1]

    assert auroc(TwoSetBenchmark(), [a, b], **cached("one-call")) == expected

    benchmark = TwoSetBenchmark()
    across = cached("across-calls")
    assert auroc(benchmark, [a], **across) + auroc(benchmark, [b], **across) == expected

    # the same model object, after its weights changed
    b.load_state_dict(a.state_dict())
    assert auroc(benchmark, [b], **across) == expected[:2]


def _smaller_dataset(benchmark, on_disk, model, monkeypatch):
    benchmark._test_ood = TensorDataset(*[t[:1] for t in benchmark._test_ood.tensors])


def _other_key(benchmark, on_disk, model, monkeypatch):
    on_disk["cache_key"] = "other"


def _other_benchmark(benchmark, on_disk, model, monkeypatch):
    benchmark.__class__ = RenamedBenchmark


def _newer_format(benchmark, on_disk, model, monkeypatch):
    monkeypatch.setattr(base_module, "_CACHE_VERSION", base_module._CACHE_VERSION + 1)


def _submodule_mode(benchmark, on_disk, model, monkeypatch):
    model.linear.train(not model.linear.training)


@pytest.mark.parametrize(
    "change",
    [_smaller_dataset, _other_key, _other_benchmark, _newer_format, _submodule_mode],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_disk_entry_is_not_reused_after(change, tmp_path, monkeypatch):
    on_disk = {"cache_dir": str(tmp_path), "cache_key": "toy"}
    model = CountingLinear()
    ToyBenchmark().evaluate(MaxSoftmax(model), **LOADER, **on_disk)
    calls = model.calls

    benchmark = ToyBenchmark()
    change(benchmark, on_disk, model, monkeypatch)
    benchmark.evaluate(MaxSoftmax(model), **LOADER, **on_disk)

    assert model.calls > calls


@pytest.mark.parametrize("corrupt", [b"", b"not a checkpoint"], ids=["empty", "garbage"])
def test_unreadable_disk_entry_is_recomputed(corrupt, tmp_path):
    on_disk = {"cache_dir": str(tmp_path), "cache_key": "toy"}
    model = CountingLinear()
    ToyBenchmark().evaluate(MaxSoftmax(model), **LOADER, **on_disk)
    next(tmp_path.glob("*.pt")).write_bytes(corrupt)

    with pytest.warns(UserWarning, match="unreadable"):
        ToyBenchmark().evaluate(MaxSoftmax(model), **LOADER, **on_disk)

    assert model.calls == 4


def test_cache_dir_alone_keeps_nothing_in_memory(tmp_path):
    on_disk = {"cache_dir": str(tmp_path / "cache"), "cache_key": "toy"}
    benchmark, model = ToyBenchmark(), CountingLinear()

    benchmark.evaluate(MaxSoftmax(model), **LOADER, **on_disk)
    shutil.rmtree(on_disk["cache_dir"])
    benchmark.evaluate(MaxSoftmax(model), **LOADER, **on_disk)

    assert model.calls == 4


@pytest.mark.parametrize("wrap", [lambda fn: fn, ParameterFree], ids=["lambda", "no-weights"])
def test_producer_without_weights_is_only_shared_within_one_call(wrap, tmp_path):
    calls = []

    def negated(x):
        calls.append(1)
        return -x.float()

    a, b = wrap(negated), wrap(lambda x: x.float())
    detectors = [SummedFeaturesDetector(p) for p in (a, a, b)]
    caching = {"cache": True, "cache_dir": str(tmp_path), "cache_key": "toy"}
    benchmark = ToyBenchmark()

    with pytest.warns(UserWarning, match="cannot be identified"):
        results = [r["AUROC"] for r in benchmark.evaluate(detectors, **LOADER, **caching)]

    assert results[0] == results[1] != results[2]
    assert len(calls) == 2

    benchmark.evaluate(detectors, **LOADER, **caching)
    assert len(calls) == 4
    assert not list(tmp_path.iterdir())
