"""
OODMetrics: the fixed collection of the commonly reported metrics.
"""

import math
import unittest
import warnings
from unittest import mock

import pytest
import torch

from pytorch_ood.metrics import AUPR, AUTC, AveragePrecision, MetricCollection, OODMetrics
from pytorch_ood.metrics import functional as F
from tests.helpers import requires


def _separated(n=10):
    scores = torch.zeros(n)
    labels = torch.zeros(n).long()
    scores[n // 2 :] = 1
    labels[n // 2 :] = -1
    return scores, labels


class TestOODMetrics(unittest.TestCase):
    def test_perfect_separation(self):
        scores, labels = _separated()
        result = OODMetrics().update(scores, labels).compute()
        assert result == {
            "AUROC": 1.0,
            "AUTC": 0.0,
            "AUPR-IN": 1.0,
            "AUPR-OUT": 1.0,
            "FPR95TPR": 0.0,
        }
        assert all(isinstance(v, float) for v in result.values())

    def test_does_not_include_average_precision(self):
        scores, labels = _separated()
        result = OODMetrics().update(scores, labels, labels.clamp(min=0)).compute()
        assert tuple(result) == ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR", "ACC")

    def test_values_match_functional(self):
        g = torch.Generator().manual_seed(0)
        labels = torch.randint(-1, 10, (5000,), generator=g)
        scores = (torch.randn(5000, generator=g) + (labels < 0).float()) * 1e4
        predictions = torch.randint(0, 10, (5000,), generator=g)
        result = OODMetrics(fpr_at=0.9).update(scores, labels, predictions).compute()
        assert result["AUROC"] == float(F.auroc(scores, labels))
        assert result["AUTC"] == float(F.autc(scores, labels))
        assert result["AUPR-IN"] == float(F.aupr(scores, labels, positive="id"))
        assert result["AUPR-OUT"] == float(F.aupr(scores, labels, positive="ood"))
        assert result["FPR90TPR"] == float(F.fpr_at_tpr(scores, labels, 0.9))
        assert result["ACC"] == float(F.accuracy(predictions, labels))

    def test_accuracy(self):
        scores, labels = _separated()
        predictions = torch.zeros(10).long()
        # predictions of OOD samples are irrelevant
        predictions[7] = 3
        assert OODMetrics().update(scores, labels, predictions).compute()["ACC"] == 1.0
        predictions[:2] = 1
        assert OODMetrics().update(scores, labels, predictions).compute()["ACC"] == pytest.approx(
            3 / 5, abs=5e-8
        )

    def test_no_accuracy_without_predictions(self):
        scores, labels = _separated()
        assert "ACC" not in OODMetrics().update(scores, labels).compute()

    def test_predictions_in_some_updates_only_raise(self):
        scores, labels = _separated()
        metrics = OODMetrics().update(scores, labels, labels.clamp(min=0))
        with pytest.raises(ValueError):
            metrics.update(scores, labels)
        metrics = OODMetrics().update(scores, labels)
        with pytest.raises(ValueError):
            metrics.update(scores, labels, labels.clamp(min=0))

    def test_predictions_after_reset(self):
        scores, labels = _separated()
        metrics = OODMetrics().update(scores, labels, labels.clamp(min=0))
        metrics.reset()
        assert "ACC" not in metrics.update(scores, labels).compute()

    def test_accuracy_without_id_samples_raises(self):
        scores, labels = _separated()
        metrics = OODMetrics(void_label=0).update(scores, labels, labels.clamp(min=0))
        with pytest.raises(ValueError):
            metrics.compute()

    def test_shape_mismatch(self):
        scores, labels = _separated()
        with pytest.raises(ValueError):
            OODMetrics().update(scores, labels, torch.zeros(5).long())
        with pytest.raises(ValueError):
            OODMetrics().update(scores, labels[:5])

    def test_only_id_or_only_ood_raises(self):
        for labels in (torch.zeros(10).long(), -torch.ones(10).long()):
            metrics = OODMetrics().update(torch.arange(10.0), labels)
            with pytest.raises(ValueError):
                metrics.compute()

    def test_inputs_of_any_shape(self):
        scores, labels = _separated(16)
        expected = OODMetrics().update(scores, labels).compute()
        assert (
            OODMetrics().update(scores.view(2, 2, 4), labels.view(2, 2, 4)).compute() == expected
        )

    def test_autc_is_nan_for_constant_scores(self):
        labels = torch.tensor([0, 0, -1, -1])
        with pytest.warns(UserWarning):
            result = OODMetrics().update(torch.ones(4), labels).compute()
        assert math.isnan(result["AUTC"])
        assert result["AUROC"] == 0.5

    def test_autc_is_nan_for_infinite_scores(self):
        labels = torch.tensor([0, 0, 0, -1, -1, -1])
        scores = torch.tensor([-float("inf"), 0.0, 1.0, 1.0, 2.0, float("inf")])
        with pytest.warns(UserWarning):
            result = OODMetrics().update(scores, labels).compute()
        assert math.isnan(result["AUTC"])
        # infinite scores rank like any other: same as finite scores in the same order
        expected = OODMetrics().update(torch.tensor([-9.0, 0, 1, 1, 2, 9]), labels).compute()
        for key in ("AUROC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"):
            assert result[key] == expected[key], key

    def test_autc_raises_for_invalid_data(self):
        # only constant and infinite scores give NaN; invalid data raises as for the others
        labels = torch.tensor([0, 0, -1, -1])
        cases = {
            "NaN": (torch.tensor([0.0, float("nan"), 1.0, 2.0]), labels),
            "only ID": (torch.ones(4), torch.zeros(4).long()),
            "only OOD": (torch.arange(4.0), -torch.ones(4).long()),
        }
        for name, (scores, y) in cases.items():
            with self.subTest(name):
                metric = AUTC().update(scores, y)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with pytest.raises(ValueError):
                        metric.compute()

    def test_no_warning_for_valid_scores(self):
        scores, labels = _separated()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OODMetrics().update(scores, labels).compute()

    def test_large_scores(self):
        # torchmetrics, used before, applied a sigmoid that saturated for such scores
        g = torch.Generator().manual_seed(1)
        labels = torch.cat([torch.zeros(300), -torch.ones(100)]).long()
        scores = (torch.randn(400, generator=g) + (labels < 0).float()).round(decimals=1)
        expected = OODMetrics().update(scores / 10, labels).compute()
        for scale, offset in ((1000.0, 0.0), (1000.0, 1e4), (1000.0, -1e4), (1e6, 0.0)):
            with self.subTest(scale=scale, offset=offset):
                result = OODMetrics().update(scores * scale + offset, labels).compute()
                for key, value in expected.items():
                    assert result[key] == pytest.approx(value, abs=5e-7), key


def _tied(seed=0, n=2000, levels=4):
    """Few-valued scores, where AP and the trapezoidal AUPR differ."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(-1, 5, (n,), generator=g)
    scores = torch.randint(0, levels, (n,), generator=g).float() + (labels < 0).float()
    return scores, labels


class TestAveragePrecision(unittest.TestCase):
    def test_keys(self):
        assert AveragePrecision().keys == ("AP-OUT",)
        assert AveragePrecision("ood").keys == ("AP-OUT",)
        assert AveragePrecision("id").keys == ("AP-IN",)
        scores, labels = _tied()
        assert list(AveragePrecision("id").update(scores, labels).compute()) == ["AP-IN"]

    def test_matches_functional_batched_and_unbatched(self):
        scores, labels = _tied()
        for positive, key in (("ood", "AP-OUT"), ("id", "AP-IN")):
            with self.subTest(positive=positive):
                expected = float(F.average_precision(scores, labels, positive))
                single = AveragePrecision(positive).update(scores, labels).compute()
                batched = AveragePrecision(positive)
                # batches of 2-D inputs, which are flattened
                for s, y in zip(scores.split(400), labels.split(400)):
                    batched.update(s.view(20, 20), y.view(20, 20))
                assert single == {key: expected}
                assert batched.compute() == {key: expected}
                assert isinstance(single[key], float)
                # differs from the trapezoidal AUPR for these scores
                assert expected != pytest.approx(float(F.aupr(scores, labels, positive)), abs=5e-4)

    def test_void_label(self):
        scores, labels = _tied()
        labels[::7] = 255
        keep = labels != 255
        for positive in ("ood", "id"):
            with self.subTest(positive=positive):
                metric = AveragePrecision(positive, void_label=255).update(scores, labels)
                assert metric.compute()[metric.keys[0]] == float(
                    F.average_precision(scores[keep], labels[keep], positive)
                )

    def test_invalid_parameters(self):
        for positive in ("out", "in", "OOD", None):
            with self.subTest(positive=positive):
                with pytest.raises(ValueError):
                    AveragePrecision(positive)
        with pytest.raises(ValueError):
            AveragePrecision(void_label=-1)

    def test_invalid_data_raises(self):
        labels = torch.tensor([0, 0, -1, -1])
        for name, (scores, y) in {
            "NaN": (torch.tensor([0.0, float("nan"), 1.0, 2.0]), labels),
            "only ID": (torch.arange(4.0), torch.zeros(4).long()),
            "only OOD": (torch.arange(4.0), -torch.ones(4).long()),
        }.items():
            with self.subTest(name):
                with pytest.raises(ValueError):
                    AveragePrecision().update(scores, y).compute()
        with pytest.raises(ValueError):
            AveragePrecision().compute()

    def test_in_collection_with_aupr(self):
        scores, labels = _tied()
        metrics = [AUPR("ood"), AveragePrecision("ood"), AUPR("id"), AveragePrecision("id")]
        collection = MetricCollection(metrics)
        for s, y in zip(scores.split(500), labels.split(500)):
            collection.update(scores=s, labels=y)
        with mock.patch("torch.argsort", wraps=torch.argsort) as argsort:
            result = collection.compute()
        # one curve for all four
        assert argsort.call_count == 1
        assert result == {
            "AUPR-OUT": float(F.aupr(scores, labels, "ood")),
            "AP-OUT": float(F.average_precision(scores, labels, "ood")),
            "AUPR-IN": float(F.aupr(scores, labels, "id")),
            "AP-IN": float(F.average_precision(scores, labels, "id")),
        }
        with pytest.raises(ValueError):
            MetricCollection([AveragePrecision(), AveragePrecision("ood")])

    @requires(cuda=1)
    def test_cuda(self):
        scores, labels = _tied()
        expected = AveragePrecision("id").update(scores, labels).compute()
        metric = AveragePrecision("id").update(scores.cuda(), labels)
        assert all(t.device.type == "cuda" for t in metric._buffers["scores"])
        assert all(t.device.type == "cuda" for t in metric._buffers["labels"])
        assert metric._compute()["AP-IN"].device.type == "cuda"
        assert metric.compute()["AP-IN"] == pytest.approx(expected["AP-IN"], abs=5e-13)
