import tempfile
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import FeaturesDetector
from pytorch_ood.benchmark import Benchmark
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


def test_cache_file_path_distinguishes_bound_method_producers():
    benchmark = ToyBenchmark()
    model = CountingFeatures()

    forward_path = benchmark._cache_file_path(
        cache_dir="/tmp",
        split="eval",
        dataset_name="ToyOOD",
        representation="features",
        producer=model.forward,
        cache_key="toy-model-v1",
    )
    pre_pool_path = benchmark._cache_file_path(
        cache_dir="/tmp",
        split="eval",
        dataset_name="ToyOOD",
        representation="features",
        producer=model.feature_maps,
        cache_key="toy-model-v1",
    )

    assert forward_path != pre_pool_path


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


def test_benchmark_reuses_in_memory_cache_across_calls():
    benchmark = ToyBenchmark()
    model = CountingLinear()

    benchmark.evaluate(
        MaxSoftmax(model),
        loader_kwargs={"batch_size": 2},
        cache=True,
    )
    calls_after_first = model.calls

    benchmark.evaluate(
        EnergyBased(model),
        loader_kwargs={"batch_size": 2},
        cache=True,
    )

    assert calls_after_first == 2
    assert model.calls == calls_after_first


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


def test_benchmark_reuses_feature_cache_across_calls():
    benchmark = ToyBenchmark()
    model = CountingFeatures()

    benchmark.evaluate(
        SummedFeaturesDetector(model),
        loader_kwargs={"batch_size": 2},
        cache=True,
    )
    calls_after_first = model.calls

    benchmark.evaluate(
        SummedFeaturesDetector(model),
        loader_kwargs={"batch_size": 2},
        cache=True,
    )

    assert calls_after_first == 2
    assert model.calls == calls_after_first


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
