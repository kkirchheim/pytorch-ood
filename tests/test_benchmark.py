import tempfile
import unittest
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import FeaturesDetector
from pytorch_ood.benchmark import Benchmark
from pytorch_ood.detector import EnergyBased, Mahalanobis, MaxSoftmax


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


class SummedFeaturesDetector(FeaturesDetector):
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.predict_features(self.model(x))

    def predict_features(self, x):
        return -x.sum(dim=1)


class BenchmarkCachingTest(unittest.TestCase):
    def test_benchmark_evaluate_single_detector_keeps_output_shape(self):
        benchmark = ToyBenchmark()
        detector = MaxSoftmax(CountingLinear())

        results = benchmark.evaluate(detector, loader_kwargs={"batch_size": 2})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["Dataset"], "ToyOOD")
        self.assertNotIn("Detector", results[0])

    def test_benchmark_evaluate_list_shares_logits_within_call(self):
        benchmark = ToyBenchmark()
        model = CountingLinear()

        results = benchmark.evaluate(
            [MaxSoftmax(model), EnergyBased(model)],
            loader_kwargs={"batch_size": 2},
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            {result["Detector"] for result in results},
            {"MaxSoftmax", "EnergyBased"},
        )
        self.assertEqual(model.calls, 2)

    def test_benchmark_reuses_in_memory_cache_across_calls(self):
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

        self.assertEqual(calls_after_first, 2)
        self.assertEqual(model.calls, calls_after_first)

    def test_benchmark_reuses_disk_cache_across_benchmark_instances(self):
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

            self.assertEqual(model.calls, 2)
            self.assertEqual(len(caught), 1)
            self.assertIn("cache_key", str(caught[0].message))

            second_benchmark = ToyBenchmark()
            second_model = CountingLinear()

            second_benchmark.evaluate(
                EnergyBased(second_model),
                loader_kwargs={"batch_size": 2},
                cache=True,
                cache_dir=tmpdir,
                cache_key="toy-model-v1",
            )

            self.assertEqual(second_model.calls, 0)

    def test_benchmark_reuses_feature_cache_across_calls(self):
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

        self.assertEqual(calls_after_first, 2)
        self.assertEqual(model.calls, calls_after_first)

    def test_benchmark_falls_back_for_mahalanobis_with_input_preprocessing(self):
        benchmark = ToyBenchmark()
        model = CountingFeatures()
        detector = Mahalanobis(model=model, eps=0.1)
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

        self.assertGreater(calls_after_first_eval, calls_after_fit)
        self.assertGreater(model.calls, calls_after_first_eval)


if __name__ == "__main__":
    unittest.main()
