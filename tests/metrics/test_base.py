"""
The metric interface: buffering, devices, void labels, and collections.
"""

import unittest
from unittest import mock

import torch

from src.pytorch_ood.metrics import (
    AUPR,
    AUROC,
    AUTC,
    Accuracy,
    FPRAtTPR,
    MetricCollection,
    OODMetrics,
)
from src.pytorch_ood.metrics import base as base_module
from src.pytorch_ood.metrics import functional as F

CUDA = torch.cuda.is_available()


def _batches(seed=0, n_batches=5, size=64):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        labels = torch.randint(-1, 5, (size,), generator=g)
        scores = torch.randn(size, generator=g) + (labels < 0).float()
        predictions = torch.randint(0, 5, (size,), generator=g)
        out.append((scores, labels, predictions))
    return out


def _cat(batches):
    return [torch.cat(parts) for parts in zip(*batches)]


class TestMetricUpdates(unittest.TestCase):
    def test_streaming_equals_single_update(self):
        batches = _batches()
        scores, labels, predictions = _cat(batches)
        for metric_cls in (AUROC, AUTC, lambda: AUPR("id"), lambda: AUPR("ood"), FPRAtTPR):
            batched = metric_cls()
            for s, y, _ in batches:
                batched.update(s, y)
            single = metric_cls().update(scores, labels)
            self.assertEqual(batched.compute(), single.compute())
        batched = Accuracy()
        for _, y, p in batches:
            batched.update(p, y)
        self.assertEqual(batched.compute(), Accuracy().update(predictions, labels).compute())

    def test_matches_functional(self):
        scores, labels, predictions = _cat(_batches())
        self.assertEqual(
            AUROC().update(scores, labels).compute()["AUROC"], float(F.auroc(scores, labels))
        )
        self.assertEqual(
            AUPR("id").update(scores, labels).compute()["AUPR-IN"],
            float(F.aupr(scores, labels, positive="id")),
        )
        self.assertEqual(
            FPRAtTPR(0.8).update(scores, labels).compute()["FPR80TPR"],
            float(F.fpr_at_tpr(scores, labels, 0.8)),
        )
        self.assertEqual(
            AUTC().update(scores, labels).compute()["AUTC"], float(F.autc(scores, labels))
        )
        self.assertEqual(
            Accuracy().update(predictions, labels).compute()["ACC"],
            float(F.accuracy(predictions, labels)),
        )

    def test_keyword_inputs(self):
        scores, labels, predictions = _cat(_batches())
        self.assertEqual(
            AUROC().update(scores=scores, labels=labels).compute(),
            AUROC().update(scores, labels).compute(),
        )
        self.assertEqual(
            Accuracy().update(labels=labels, predictions=predictions).compute(),
            Accuracy().update(predictions, labels).compute(),
        )

    def test_invalid_inputs(self):
        scores, labels, _ = _cat(_batches())
        with self.assertRaises(ValueError):
            AUROC().update(scores)
        with self.assertRaises(TypeError):
            AUROC().update(scores, labels, labels)
        with self.assertRaises(TypeError):
            AUROC().update(scores, scores=scores)
        with self.assertRaises(ValueError):
            AUROC().update(scores, labels[:-1])
        with self.assertRaises(TypeError):
            AUROC().update(scores.tolist(), labels)

    def test_compute_without_data_raises(self):
        for metric in (AUROC(), AUTC(), AUPR(), FPRAtTPR(), Accuracy(), OODMetrics()):
            with self.subTest(type(metric).__name__):
                with self.assertRaises(ValueError):
                    metric.compute()

    def test_empty_update_is_ignored(self):
        scores, labels, _ = _cat(_batches())
        metric = AUROC().update(torch.zeros(0), torch.zeros(0).long()).update(scores, labels)
        self.assertEqual(metric.compute(), AUROC().update(scores, labels).compute())

    def test_compute_is_repeatable_and_update_continues(self):
        batches = _batches()
        metric = OODMetrics()
        metric.update(*batches[0])
        first = metric.compute()
        self.assertEqual(metric.compute(), first)
        for batch in batches[1:]:
            metric.update(*batch)
        self.assertEqual(metric.compute(), OODMetrics().update(*_cat(batches)).compute())

    def test_reset(self):
        batches = _batches()
        for metric in (AUROC(), Accuracy(), OODMetrics()):
            with self.subTest(type(metric).__name__):
                args = (
                    (lambda b: (b[2], b[1])) if isinstance(metric, Accuracy) else (lambda b: b[:2])
                )
                metric.update(*args(batches[0]))
                metric.reset()
                with self.assertRaises(ValueError):
                    metric.compute()
                metric.update(*args(batches[1]))
                fresh = type(metric)().update(*args(batches[1]))
                self.assertEqual(metric.compute(), fresh.compute())

    def test_inputs_are_not_modified_and_detached(self):
        scores = torch.randn(10, requires_grad=True)
        labels = torch.tensor([0] * 5 + [-1] * 5)
        metric = AUROC().update(scores, labels)
        self.assertFalse(metric._buffers["scores"][0].requires_grad)
        self.assertIsInstance(metric.compute()["AUROC"], float)

    def test_labels_and_predictions_stored_as_int32(self):
        scores, labels, predictions = _cat(_batches())
        collection = OODMetrics().update(scores, labels.double(), predictions)
        self.assertEqual(collection._buffers["labels"][0].dtype, torch.int32)
        self.assertEqual(collection._buffers["scores"][0].dtype, scores.dtype)
        # class indices beyond the range of int16: the wrong prediction equals the label modulo
        # 2^16, so a 16-bit cast would count it as correct
        labels = torch.tensor([70000, 40000])
        predictions = torch.tensor([70000 - 2**16, 40000])
        self.assertEqual(Accuracy().update(predictions, labels).compute()["ACC"], 0.5)

    def test_accuracy_counts_beyond_float32_precision(self):
        # float32 counts are exact only up to 2^24
        n = 2**24 + 3
        labels = torch.zeros(n, dtype=torch.int32)
        predictions = torch.zeros(n, dtype=torch.int32)
        predictions[0] = 1
        metric = Accuracy()
        for _ in range(2):
            metric.update(predictions, labels)
        self.assertEqual(int(metric._correct), 2 * (n - 1))
        self.assertEqual(int(metric._total), 2 * n)
        self.assertEqual(metric.compute()["ACC"], (n - 1) / n)

    def test_buffered_inputs_do_not_share_memory_with_inputs(self):
        scores, labels, predictions = _cat(_batches())
        for metric, args in (
            (AUROC(), (scores, labels)),
            (OODMetrics(), (scores, labels, predictions)),
        ):
            with self.subTest(type(metric).__name__):
                expected = type(metric)().update(*args).compute()
                given = [a.clone() for a in args]
                metric.update(*given)
                for tensor in given:
                    tensor.fill_(0)
                self.assertEqual(metric.compute(), expected)

    def test_device_and_void_label_are_keyword_only(self):
        for metric_cls in (AUROC, AUTC, Accuracy, OODMetrics):
            with self.subTest(metric_cls.__name__):
                with self.assertRaises(TypeError):
                    metric_cls("cpu")
        with self.assertRaises(TypeError):
            AUPR("id", "cpu")
        with self.assertRaises(TypeError):
            FPRAtTPR(0.95, "cpu")
        with self.assertRaises(TypeError):
            MetricCollection([AUROC()], "cpu")

    def test_unknown_input_raises(self):
        scores, labels, _ = _cat(_batches())
        with self.assertRaises(TypeError):
            AUROC().update(scores=scores, labels=labels, logits=scores)
        with self.assertRaises(TypeError):
            AUROC().update(scores, labels, scores)

    def test_float_labels(self):
        scores, labels, _ = _cat(_batches())
        self.assertEqual(
            AUROC().update(scores, labels.float()).compute(),
            AUROC().update(scores, labels).compute(),
        )

    def test_nan_scores_raise(self):
        metric = OODMetrics().update(torch.tensor([0.0, float("nan")]), torch.tensor([0, -1]))
        with self.assertRaises(ValueError):
            metric.compute()


class TestVoidLabel(unittest.TestCase):
    def _batch(self):
        # ID label 1 with low scores, OOD with high scores, void label 0 with the highest scores
        scores = torch.tensor([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1])
        labels = torch.tensor([1, 1, 1, -1, -1, -1, 0, 0])
        predictions = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1])  # void samples misclassified
        return scores, labels, predictions

    def test_excludes_void_entries(self):
        scores, labels, predictions = self._batch()
        result = OODMetrics(void_label=0).update(scores, labels, predictions).compute()
        self.assertEqual(result["AUROC"], 1.0)
        self.assertEqual(result["FPR95TPR"], 0.0)
        self.assertEqual(result["ACC"], 1.0)
        # without the void label, the void samples count as ID with the highest scores
        self.assertLess(OODMetrics().update(scores, labels).compute()["AUROC"], 1.0)

    def test_single_metrics(self):
        scores, labels, predictions = self._batch()
        self.assertEqual(AUROC(void_label=0).update(scores, labels).compute()["AUROC"], 1.0)
        self.assertEqual(Accuracy(void_label=0).update(predictions, labels).compute()["ACC"], 1.0)

    def test_only_void_raises(self):
        metric = OODMetrics(void_label=3).update(torch.zeros(4), torch.full((4,), 3))
        with self.assertRaises(ValueError):
            metric.compute()

    def test_only_one_class_after_removing_void_raises(self):
        metric = OODMetrics(void_label=3).update(torch.arange(4.0), torch.tensor([3, 3, 0, 0]))
        with self.assertRaises(ValueError):
            metric.compute()

    def test_negative_void_label_raises(self):
        for cls in (AUROC, Accuracy, OODMetrics):
            with self.assertRaises(ValueError):
                cls(void_label=-1)


class TestDevices(unittest.TestCase):
    def test_default_keeps_device_of_first_input(self):
        scores, labels, _ = _cat(_batches())
        metric = AUROC().update(scores, labels)
        self.assertEqual(metric._buffers["scores"][0].device.type, "cpu")

    def test_explicit_cpu(self):
        scores, labels, _ = _cat(_batches())
        metric = AUROC(device="cpu").update(scores, labels)
        self.assertEqual(metric._buffers["labels"][0].device.type, "cpu")

    @unittest.skipUnless(CUDA, "requires CUDA")
    def test_mixed_devices_in_one_update(self):
        scores, labels, predictions = _cat(_batches())
        expected = OODMetrics().update(scores, labels, predictions).compute()
        metric = OODMetrics().update(scores.cuda(), labels, predictions)
        self.assertEqual(metric._buffers["labels"][0].device.type, "cuda")
        result = metric.compute()
        for key in expected:
            self.assertAlmostEqual(result[key], expected[key], places=12, msg=key)

    @unittest.skipUnless(CUDA, "requires CUDA")
    def test_devices_across_updates(self):
        batches = _batches()
        metric = OODMetrics()
        metric.update(*(t.cuda() for t in batches[0]))
        for batch in batches[1:]:
            metric.update(*batch)
        self.assertTrue(all(t.device.type == "cuda" for t in metric._buffers["scores"]))
        expected = OODMetrics().update(*_cat(batches)).compute()
        for key, value in metric.compute().items():
            self.assertAlmostEqual(value, expected[key], places=12, msg=key)

    @unittest.skipUnless(CUDA, "requires CUDA")
    def test_explicit_device_wins(self):
        scores, labels, _ = _cat(_batches())
        metric = AUROC(device="cpu").update(scores.cuda(), labels.cuda())
        self.assertEqual(metric._buffers["scores"][0].device.type, "cpu")
        metric = AUROC(device="cuda").update(scores, labels)
        self.assertEqual(metric._buffers["scores"][0].device.type, "cuda")

    @unittest.skipUnless(CUDA, "requires CUDA")
    def test_reset_fixes_the_device_again(self):
        scores, labels, _ = _cat(_batches())
        metric = AUROC().update(scores.cuda(), labels)
        metric.reset()
        metric.update(scores, labels)
        self.assertEqual(metric._buffers["scores"][0].device.type, "cpu")

    @unittest.skipUnless(CUDA, "requires CUDA")
    def test_accuracy_on_cuda(self):
        _, labels, predictions = _cat(_batches())
        metric = Accuracy().update(predictions.cuda(), labels)
        self.assertEqual(metric._correct.device.type, "cuda")
        self.assertEqual(metric.compute(), Accuracy().update(predictions, labels).compute())


class TestCollection(unittest.TestCase):
    def test_results_equal_single_metrics(self):
        scores, labels, predictions = _cat(_batches())
        metrics = [AUROC(), AUPR("ood"), FPRAtTPR(0.9), AUTC(), Accuracy()]
        collection = MetricCollection(metrics)
        collection.update(scores=scores, labels=labels, predictions=predictions)
        expected = {}
        for metric_cls, args in (
            (AUROC, ()),
            (lambda: AUPR("ood"), ()),
            (lambda: FPRAtTPR(0.9), ()),
            (AUTC, ()),
        ):
            expected.update(metric_cls().update(scores, labels).compute())
        expected.update(Accuracy().update(predictions, labels).compute())
        self.assertEqual(collection.compute(), expected)
        self.assertEqual(
            list(collection.compute()), ["AUROC", "AUPR-OUT", "FPR90TPR", "AUTC", "ACC"]
        )

    def test_each_input_stored_once_per_update(self):
        batches = _batches(n_batches=3, size=64)
        collection = OODMetrics()
        for scores, labels, predictions in batches:
            collection.update(scores, labels, predictions)
        for name in ("scores", "labels"):
            self.assertEqual(len(collection._buffers[name]), 3)
            self.assertEqual(sum(t.numel() for t in collection._buffers[name]), 3 * 64)
        first = collection.compute()
        # the concatenation is kept for later calls of compute
        for name in ("scores", "labels"):
            self.assertEqual(len(collection._buffers[name]), 1)
            self.assertEqual(collection._buffers[name][0].numel(), 3 * 64)
        self.assertEqual(collection.compute(), first)

    def test_reset_resets_members(self):
        scores, labels, predictions = _cat(_batches())
        wrong = torch.where(labels >= 0, labels + 1, labels)
        for collection, args in (
            (OODMetrics(), lambda p: (scores, labels, p)),
            (MetricCollection([Accuracy()]), None),
        ):
            with self.subTest(type(collection).__name__):
                if args is None:
                    collection.update(predictions=wrong, labels=labels)
                    collection.reset()
                    collection.update(predictions=labels.clamp(min=0), labels=labels)
                else:
                    collection.update(*args(wrong))
                    collection.reset()
                    collection.update(*args(labels.clamp(min=0)))
                self.assertEqual(collection.compute()["ACC"], 1.0)

    def test_failed_update_changes_nothing(self):
        scores, labels, predictions = _cat(_batches())
        collection = OODMetrics()
        with self.assertRaises(ValueError):
            collection.update(scores, labels, predictions[:-1])
        # the failed call must not have committed to passing predictions
        result = collection.update(scores, labels).compute()
        self.assertNotIn("ACC", result)
        self.assertEqual(result, OODMetrics().update(scores, labels).compute())

    def test_inputs_buffered_once(self):
        scores, labels, predictions = _cat(_batches())
        collection = MetricCollection([AUROC(), AUPR("id"), AUPR("ood"), FPRAtTPR(), Accuracy()])
        collection.update(scores=scores, labels=labels, predictions=predictions)
        # only the inputs of the buffered metrics, each once; predictions only feed the accuracy
        self.assertEqual(set(collection._buffers), {"scores", "labels"})
        for metric in collection.metrics:
            if hasattr(metric, "_buffers"):
                self.assertTrue(all(not values for values in metric._buffers.values()))

    def test_curve_computed_once(self):
        # every curve sorts the scores, so one sort means one curve, whichever code path
        # computes it
        scores, labels, predictions = _cat(_batches())
        collection = OODMetrics().update(scores, labels, predictions)
        with mock.patch("torch.argsort", wraps=torch.argsort) as argsort:
            collection.compute()
        self.assertEqual(argsort.call_count, 1)
        # AUROC and FPR95TPR share the ROC curve, which is computed on first use
        counts = F._counts(scores, labels)
        self.assertIs(F._roc(counts), F._roc(counts))

    def test_streaming_members_store_counts_only(self):
        _, labels, predictions = _cat(_batches())
        collection = MetricCollection([Accuracy()])
        for _ in range(3):
            collection.update(predictions=predictions, labels=labels)
        self.assertEqual(collection._buffers, {})
        self.assertEqual(collection.metrics[0]._correct.numel(), 1)

    def test_missing_input_raises(self):
        scores, labels, _ = _cat(_batches())
        collection = MetricCollection([AUROC(), Accuracy()])
        with self.assertRaises(ValueError):
            collection.update(scores=scores, labels=labels)

    def test_invalid_construction(self):
        with self.assertRaises(ValueError):
            MetricCollection([])
        with self.assertRaises(ValueError):
            MetricCollection([AUROC(), AUROC()])
        with self.assertRaises(ValueError):
            MetricCollection([FPRAtTPR(0.95), FPRAtTPR(0.95)])
        with self.assertRaises(ValueError):
            MetricCollection([MetricCollection([AUROC()])])
        with self.assertRaises(ValueError):
            MetricCollection([AUROC(device="cpu")])
        with self.assertRaises(ValueError):
            MetricCollection([AUROC(void_label=1)])

    def test_invalid_update(self):
        scores, labels, _ = _cat(_batches())
        collection = MetricCollection([AUROC()])
        with self.assertRaises(TypeError):
            collection.update(scores, labels)
        with self.assertRaises(TypeError):
            collection.update(scores=scores, labels=labels, logits=scores)
        with self.assertRaises(ValueError):
            collection.update(scores=scores, labels=labels[:-1])

    def test_defaults(self):
        self.assertEqual(AUPR().keys, ("AUPR-OUT",))
        self.assertEqual(FPRAtTPR().keys, ("FPR95TPR",))

    def test_keys(self):
        self.assertEqual(FPRAtTPR(0.95).keys, ("FPR95TPR",))
        self.assertEqual(FPRAtTPR(0.9).keys, ("FPR90TPR",))
        self.assertEqual(FPRAtTPR(0.925).keys, ("FPR92.5TPR",))
        self.assertEqual(AUPR("id").keys, ("AUPR-IN",))
        self.assertEqual(AUPR("ood").keys, ("AUPR-OUT",))
        self.assertEqual(
            OODMetrics().keys, ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR", "ACC")
        )
        self.assertEqual(OODMetrics(fpr_at=0.9).keys[4], "FPR90TPR")

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            AUPR("out")
        for tpr in (-0.1, 1.1, 95):
            with self.assertRaises(ValueError):
                FPRAtTPR(tpr)
            with self.assertRaises(ValueError):
                OODMetrics(fpr_at=tpr)
