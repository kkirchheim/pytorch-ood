"""
OODMetrics: the fixed collection of the commonly reported metrics.
"""

import math
import unittest
import warnings

import torch

from src.pytorch_ood.metrics import OODMetrics
from src.pytorch_ood.metrics import functional as F


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
        self.assertEqual(
            result,
            {"AUROC": 1.0, "AUTC": 0.0, "AUPR-IN": 1.0, "AUPR-OUT": 1.0, "FPR95TPR": 0.0},
        )
        self.assertTrue(all(isinstance(v, float) for v in result.values()))

    def test_values_match_functional(self):
        g = torch.Generator().manual_seed(0)
        labels = torch.randint(-1, 10, (5000,), generator=g)
        scores = (torch.randn(5000, generator=g) + (labels < 0).float()) * 1e4
        predictions = torch.randint(0, 10, (5000,), generator=g)
        result = OODMetrics(fpr_at=0.9).update(scores, labels, predictions).compute()
        self.assertEqual(result["AUROC"], float(F.auroc(scores, labels)))
        self.assertEqual(result["AUTC"], float(F.autc(scores, labels)))
        self.assertEqual(result["AUPR-IN"], float(F.aupr(scores, labels, positive="id")))
        self.assertEqual(result["AUPR-OUT"], float(F.aupr(scores, labels, positive="ood")))
        self.assertEqual(result["FPR90TPR"], float(F.fpr_at_tpr(scores, labels, 0.9)))
        self.assertEqual(result["ACC"], float(F.accuracy(predictions, labels)))

    def test_accuracy(self):
        scores, labels = _separated()
        predictions = torch.zeros(10).long()
        # predictions of OOD samples are irrelevant
        predictions[7] = 3
        self.assertEqual(OODMetrics().update(scores, labels, predictions).compute()["ACC"], 1.0)
        predictions[:2] = 1
        self.assertAlmostEqual(
            OODMetrics().update(scores, labels, predictions).compute()["ACC"], 3 / 5
        )

    def test_no_accuracy_without_predictions(self):
        scores, labels = _separated()
        self.assertNotIn("ACC", OODMetrics().update(scores, labels).compute())

    def test_predictions_in_some_updates_only_raise(self):
        scores, labels = _separated()
        metrics = OODMetrics().update(scores, labels, labels.clamp(min=0))
        with self.assertRaises(ValueError):
            metrics.update(scores, labels)
        metrics = OODMetrics().update(scores, labels)
        with self.assertRaises(ValueError):
            metrics.update(scores, labels, labels.clamp(min=0))

    def test_predictions_after_reset(self):
        scores, labels = _separated()
        metrics = OODMetrics().update(scores, labels, labels.clamp(min=0))
        metrics.reset()
        self.assertNotIn("ACC", metrics.update(scores, labels).compute())

    def test_accuracy_without_id_samples_raises(self):
        scores, labels = _separated()
        metrics = OODMetrics(void_label=0).update(scores, labels, labels.clamp(min=0))
        with self.assertRaises(ValueError):
            metrics.compute()

    def test_shape_mismatch(self):
        scores, labels = _separated()
        with self.assertRaises(ValueError):
            OODMetrics().update(scores, labels, torch.zeros(5).long())
        with self.assertRaises(ValueError):
            OODMetrics().update(scores, labels[:5])

    def test_only_id_or_only_ood_raises(self):
        for labels in (torch.zeros(10).long(), -torch.ones(10).long()):
            metrics = OODMetrics().update(torch.arange(10.0), labels)
            with self.assertRaises(ValueError):
                metrics.compute()

    def test_inputs_of_any_shape(self):
        scores, labels = _separated(16)
        expected = OODMetrics().update(scores, labels).compute()
        self.assertEqual(
            OODMetrics().update(scores.view(2, 2, 4), labels.view(2, 2, 4)).compute(), expected
        )

    def test_autc_is_nan_for_constant_scores(self):
        labels = torch.tensor([0, 0, -1, -1])
        with self.assertWarns(UserWarning):
            result = OODMetrics().update(torch.ones(4), labels).compute()
        self.assertTrue(math.isnan(result["AUTC"]))
        self.assertEqual(result["AUROC"], 0.5)

    def test_autc_is_nan_for_infinite_scores(self):
        labels = torch.tensor([0, 0, 0, -1, -1, -1])
        scores = torch.tensor([-float("inf"), 0.0, 1.0, 1.0, 2.0, float("inf")])
        with self.assertWarns(UserWarning):
            result = OODMetrics().update(scores, labels).compute()
        self.assertTrue(math.isnan(result["AUTC"]))
        # infinite scores rank like any other: same as finite scores in the same order
        expected = OODMetrics().update(torch.tensor([-9.0, 0, 1, 1, 2, 9]), labels).compute()
        for key in ("AUROC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"):
            self.assertEqual(result[key], expected[key], key)

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
                    self.assertAlmostEqual(result[key], value, places=6, msg=key)
