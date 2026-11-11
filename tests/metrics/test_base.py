"""
The metric interface: buffering, devices, void labels, and collections.
"""

from unittest import mock

import pytest
import torch

from pytorch_ood.metrics import (
    AUPR,
    AUROC,
    AUTC,
    Accuracy,
    AveragePrecision,
    FPRAtTPR,
    MetricCollection,
    OODMetrics,
)
from pytorch_ood.metrics import functional as F
from tests.helpers import requires


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


def test_streaming_equals_single_update():
    batches = _batches()
    scores, labels, predictions = _cat(batches)
    for metric_cls in (
        AUROC,
        AUTC,
        lambda: AUPR("id"),
        lambda: AUPR("ood"),
        lambda: AveragePrecision("id"),
        lambda: AveragePrecision("ood"),
        FPRAtTPR,
    ):
        batched = metric_cls()
        for s, y, _ in batches:
            batched.update(s, y)
        single = metric_cls().update(scores, labels)
        assert batched.compute() == single.compute()
    batched = Accuracy()
    for _, y, p in batches:
        batched.update(p, y)
    assert batched.compute() == Accuracy().update(predictions, labels).compute()


def test_matches_functional():
    scores, labels, predictions = _cat(_batches())
    assert AUROC().update(scores, labels).compute()["AUROC"] == float(F.auroc(scores, labels))
    assert AUPR("id").update(scores, labels).compute()["AUPR-IN"] == float(
        F.aupr(scores, labels, positive="id")
    )
    assert AveragePrecision("id").update(scores, labels).compute()["AP-IN"] == float(
        F.average_precision(scores, labels, positive="id")
    )
    assert AveragePrecision("ood").update(scores, labels).compute()["AP-OUT"] == float(
        F.average_precision(scores, labels, positive="ood")
    )
    assert FPRAtTPR(0.8).update(scores, labels).compute()["FPR80TPR"] == float(
        F.fpr_at_tpr(scores, labels, 0.8)
    )
    assert AUTC().update(scores, labels).compute()["AUTC"] == float(F.autc(scores, labels))
    assert Accuracy().update(predictions, labels).compute()["ACC"] == float(
        F.accuracy(predictions, labels)
    )


def test_keyword_inputs():
    scores, labels, predictions = _cat(_batches())
    assert (
        AUROC().update(scores=scores, labels=labels).compute()
        == AUROC().update(scores, labels).compute()
    )
    assert (
        Accuracy().update(labels=labels, predictions=predictions).compute()
        == Accuracy().update(predictions, labels).compute()
    )


def test_invalid_inputs():
    scores, labels, _ = _cat(_batches())
    with pytest.raises(TypeError):
        AUROC().update(scores)
    with pytest.raises(ValueError):
        AUROC().update(scores, None)
    with pytest.raises(TypeError):
        AUROC().update(scores, labels, labels)
    with pytest.raises(TypeError):
        AUROC().update(scores=scores, labels=labels, logits=scores)
    with pytest.raises(TypeError):
        AUROC().update(scores, scores=scores)
    with pytest.raises(ValueError):
        AUROC().update(scores, labels[:-1])
    with pytest.raises(TypeError):
        AUROC().update(scores.tolist(), labels)


@pytest.mark.parametrize(
    "metric",
    (
        AUROC(),
        AUTC(),
        AUPR(),
        AveragePrecision(),
        FPRAtTPR(),
        Accuracy(),
        OODMetrics(),
    ),
)
def test_compute_without_data_raises(metric):
    with pytest.raises(ValueError):
        metric.compute()


def test_empty_update_is_ignored():
    scores, labels, _ = _cat(_batches())
    metric = AUROC().update(torch.zeros(0), torch.zeros(0).long()).update(scores, labels)
    assert metric.compute() == AUROC().update(scores, labels).compute()


def test_compute_is_repeatable_and_update_continues():
    batches = _batches()
    metric = OODMetrics()
    metric.update(*batches[0])
    first = metric.compute()
    assert metric.compute() == first
    for batch in batches[1:]:
        metric.update(*batch)
    assert metric.compute() == OODMetrics().update(*_cat(batches)).compute()


@pytest.mark.parametrize("metric", (AUROC(), AveragePrecision(), Accuracy(), OODMetrics()))
def test_reset(metric):
    batches = _batches()
    args = (lambda b: (b[2], b[1])) if isinstance(metric, Accuracy) else (lambda b: b[:2])
    metric.update(*args(batches[0]))
    metric.reset()
    with pytest.raises(ValueError):
        metric.compute()
    metric.update(*args(batches[1]))
    fresh = type(metric)().update(*args(batches[1]))
    assert metric.compute() == fresh.compute()


def test_inputs_are_not_modified_and_detached():
    scores = torch.randn(10, requires_grad=True)
    labels = torch.tensor([0] * 5 + [-1] * 5)
    metric = AUROC().update(scores, labels)
    assert not metric._buffers["scores"][0].requires_grad
    assert isinstance(metric.compute()["AUROC"], float)


def test_labels_and_predictions_stored_as_int32():
    scores, labels, predictions = _cat(_batches())
    collection = OODMetrics().update(scores, labels.double(), predictions)
    assert collection._buffers["labels"][0].dtype == torch.int32
    assert collection._buffers["scores"][0].dtype == scores.dtype
    # class indices beyond the range of int16: the wrong prediction equals the label modulo
    # 2^16, so a 16-bit cast would count it as correct
    labels = torch.tensor([70000, 40000])
    predictions = torch.tensor([70000 - 2**16, 40000])
    assert Accuracy().update(predictions, labels).compute()["ACC"] == 0.5


def test_accuracy_counts_beyond_float32_precision():
    # float32 counts are exact only up to 2^24
    n = 2**24 + 3
    labels = torch.zeros(n, dtype=torch.int32)
    predictions = torch.zeros(n, dtype=torch.int32)
    predictions[0] = 1
    metric = Accuracy()
    for _ in range(2):
        metric.update(predictions, labels)
    assert int(metric._correct) == 2 * (n - 1)
    assert int(metric._total) == 2 * n
    assert metric.compute()["ACC"] == (n - 1) / n


def test_buffered_inputs_do_not_share_memory_with_inputs():
    scores, labels, predictions = _cat(_batches())
    for metric, args in (
        (AUROC(), (scores, labels)),
        (OODMetrics(), (scores, labels, predictions)),
    ):
        expected = type(metric)().update(*args).compute()
        given = [a.clone() for a in args]
        metric.update(*given)
        for tensor in given:
            tensor.fill_(0)
        assert metric.compute() == expected


def test_device_and_void_label_are_keyword_only():
    for metric_cls in (AUROC, AUTC, Accuracy, OODMetrics):
        with pytest.raises(TypeError):
            metric_cls("cpu")
    with pytest.raises(TypeError):
        AUPR("id", "cpu")
    with pytest.raises(TypeError):
        AveragePrecision("id", "cpu")
    with pytest.raises(TypeError):
        FPRAtTPR(0.95, "cpu")
    with pytest.raises(TypeError):
        MetricCollection([AUROC()], "cpu")


def test_nan_scores_raise():
    metric = OODMetrics().update(torch.tensor([0.0, float("nan")]), torch.tensor([0, -1]))
    with pytest.raises(ValueError):
        metric.compute()


class TestVoidLabel:
    def _batch(self):
        # ID label 1 with low scores, OOD with high scores, void label 0 with the highest scores
        scores = torch.tensor([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1])
        labels = torch.tensor([1, 1, 1, -1, -1, -1, 0, 0])
        predictions = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1])  # void samples misclassified
        return scores, labels, predictions

    def test_excludes_void_entries(self):
        scores, labels, predictions = self._batch()
        result = OODMetrics(void_label=0).update(scores, labels, predictions).compute()
        assert result["AUROC"] == 1.0
        assert result["FPR95TPR"] == 0.0
        assert result["ACC"] == 1.0
        # without the void label, the void samples count as ID with the highest scores
        assert OODMetrics().update(scores, labels).compute()["AUROC"] < 1.0

    def test_single_metrics(self):
        scores, labels, predictions = self._batch()
        assert AUROC(void_label=0).update(scores, labels).compute()["AUROC"] == 1.0
        for positive, key in (("ood", "AP-OUT"), ("id", "AP-IN")):
            metric = AveragePrecision(positive, void_label=0).update(scores, labels)
            assert metric.compute()[key] == 1.0
            # the void samples count as ID with the highest scores otherwise
            assert AveragePrecision(positive).update(scores, labels).compute()[key] < 1
        assert Accuracy(void_label=0).update(predictions, labels).compute()["ACC"] == 1.0

    def test_only_void_raises(self):
        metric = OODMetrics(void_label=3).update(torch.zeros(4), torch.full((4,), 3))
        with pytest.raises(ValueError):
            metric.compute()

    def test_only_one_class_after_removing_void_raises(self):
        metric = OODMetrics(void_label=3).update(torch.arange(4.0), torch.tensor([3, 3, 0, 0]))
        with pytest.raises(ValueError):
            metric.compute()

    def test_negative_void_label_raises(self):
        for cls in (AUROC, AveragePrecision, Accuracy, OODMetrics):
            with pytest.raises(ValueError):
                cls(void_label=-1)


def test_default_keeps_device_of_first_input():
    scores, labels, _ = _cat(_batches())
    metric = AUROC().update(scores, labels)
    assert metric._buffers["scores"][0].device.type == "cpu"


def test_explicit_cpu():
    scores, labels, _ = _cat(_batches())
    metric = AUROC(device="cpu").update(scores, labels)
    assert metric._buffers["labels"][0].device.type == "cpu"


@requires(cuda=1)
def test_mixed_devices_in_one_update():
    scores, labels, predictions = _cat(_batches())
    expected = OODMetrics().update(scores, labels, predictions).compute()
    metric = OODMetrics().update(scores.cuda(), labels, predictions)
    assert metric._buffers["labels"][0].device.type == "cuda"
    result = metric.compute()
    for key in expected:
        assert result[key] == pytest.approx(expected[key], abs=5e-13), key


@requires(cuda=1)
def test_devices_across_updates():
    batches = _batches()
    metric = OODMetrics()
    metric.update(*(t.cuda() for t in batches[0]))
    for batch in batches[1:]:
        metric.update(*batch)
    assert all(t.device.type == "cuda" for t in metric._buffers["scores"])
    expected = OODMetrics().update(*_cat(batches)).compute()
    for key, value in metric.compute().items():
        assert value == pytest.approx(expected[key], abs=5e-13), key


@requires(cuda=1)
def test_explicit_device_wins():
    scores, labels, _ = _cat(_batches())
    metric = AUROC(device="cpu").update(scores.cuda(), labels.cuda())
    assert metric._buffers["scores"][0].device.type == "cpu"
    metric = AUROC(device="cuda").update(scores, labels)
    assert metric._buffers["scores"][0].device.type == "cuda"
    metric = AveragePrecision(device="cpu").update(scores.cuda(), labels.cuda())
    assert metric._buffers["scores"][0].device.type == "cpu"
    metric = AveragePrecision(device="cuda").update(scores, labels)
    assert metric._buffers["scores"][0].device.type == "cuda"
    assert metric._compute()["AP-OUT"].device.type == "cuda"
    assert metric.compute()["AP-OUT"] == pytest.approx(
        AveragePrecision().update(scores, labels).compute()["AP-OUT"], abs=5e-13
    )


@requires(cuda=1)
def test_reset_fixes_the_device_again():
    scores, labels, _ = _cat(_batches())
    metric = AUROC().update(scores.cuda(), labels)
    metric.reset()
    metric.update(scores, labels)
    assert metric._buffers["scores"][0].device.type == "cpu"


@requires(cuda=1)
def test_accuracy_on_cuda():
    _, labels, predictions = _cat(_batches())
    metric = Accuracy().update(predictions.cuda(), labels)
    assert metric._correct.device.type == "cuda"
    assert metric.compute() == Accuracy().update(predictions, labels).compute()


def test_results_equal_single_metrics():
    scores, labels, predictions = _cat(_batches())
    metrics = [AUROC(), AUPR("ood"), AveragePrecision("id"), FPRAtTPR(0.9), AUTC(), Accuracy()]
    collection = MetricCollection(metrics)
    collection.update(scores=scores, labels=labels, predictions=predictions)
    expected = {}
    for metric_cls, args in (
        (AUROC, ()),
        (lambda: AUPR("ood"), ()),
        (lambda: AveragePrecision("id"), ()),
        (lambda: FPRAtTPR(0.9), ()),
        (AUTC, ()),
    ):
        expected.update(metric_cls().update(scores, labels).compute())
    expected.update(Accuracy().update(predictions, labels).compute())
    assert collection.compute() == expected
    assert list(collection.compute()) == [
        "AUROC",
        "AUPR-OUT",
        "AP-IN",
        "FPR90TPR",
        "AUTC",
        "ACC",
    ]


def test_each_input_stored_once_per_update():
    batches = _batches(n_batches=3, size=64)
    collection = OODMetrics()
    for scores, labels, predictions in batches:
        collection.update(scores, labels, predictions)
    for name in ("scores", "labels"):
        assert len(collection._buffers[name]) == 3
        assert sum(t.numel() for t in collection._buffers[name]) == 3 * 64
    first = collection.compute()
    # the concatenation is kept for later calls of compute
    for name in ("scores", "labels"):
        assert len(collection._buffers[name]) == 1
        assert collection._buffers[name][0].numel() == 3 * 64
    assert collection.compute() == first


def test_reset_resets_members():
    scores, labels, predictions = _cat(_batches())
    wrong = torch.where(labels >= 0, labels + 1, labels)
    for collection, args in (
        (OODMetrics(), lambda p: (scores, labels, p)),
        (MetricCollection([Accuracy()]), None),
    ):
        if args is None:
            collection.update(predictions=wrong, labels=labels)
            collection.reset()
            collection.update(predictions=labels.clamp(min=0), labels=labels)
        else:
            collection.update(*args(wrong))
            collection.reset()
            collection.update(*args(labels.clamp(min=0)))
        assert collection.compute()["ACC"] == 1.0


def test_failed_update_changes_nothing():
    scores, labels, predictions = _cat(_batches())
    collection = OODMetrics()
    with pytest.raises(ValueError):
        collection.update(scores, labels, predictions[:-1])
    # the failed call must not have committed to passing predictions
    result = collection.update(scores, labels).compute()
    assert "ACC" not in result
    assert result == OODMetrics().update(scores, labels).compute()


def test_inputs_buffered_once():
    scores, labels, predictions = _cat(_batches())
    collection = MetricCollection(
        [AUROC(), AUPR("id"), AUPR("ood"), AveragePrecision(), FPRAtTPR(), Accuracy()]
    )
    collection.update(scores=scores, labels=labels, predictions=predictions)
    # only the inputs of the buffered metrics, each once; predictions only feed the accuracy
    assert set(collection._buffers) == {"scores", "labels"}
    for metric in collection.metrics:
        if hasattr(metric, "_buffers"):
            assert all(not values for values in metric._buffers.values())


def test_curve_computed_once():
    # every curve sorts the scores, so one sort means one curve, whichever code path
    # computes it
    scores, labels, predictions = _cat(_batches())
    collection = OODMetrics().update(scores, labels, predictions)
    with mock.patch("torch.argsort", wraps=torch.argsort) as argsort:
        collection.compute()
    assert argsort.call_count == 1
    collection = MetricCollection(
        [AUROC(), AveragePrecision("id"), AveragePrecision("ood"), AUPR("ood"), FPRAtTPR()]
    ).update(scores=scores, labels=labels)
    with mock.patch("torch.argsort", wraps=torch.argsort) as argsort:
        collection.compute()
    assert argsort.call_count == 1
    # AUROC and FPR95TPR share the ROC curve, which is computed on first use
    counts = F._counts(scores, labels)
    assert F._roc(counts) is F._roc(counts)


def test_streaming_members_store_counts_only():
    _, labels, predictions = _cat(_batches())
    collection = MetricCollection([Accuracy()])
    for _ in range(3):
        collection.update(predictions=predictions, labels=labels)
    assert collection._buffers == {}
    assert collection.metrics[0]._correct.numel() == 1


def test_missing_input_raises():
    scores, labels, _ = _cat(_batches())
    collection = MetricCollection([AUROC(), Accuracy()])
    with pytest.raises(ValueError):
        collection.update(scores=scores, labels=labels)


def test_invalid_construction():
    with pytest.raises(ValueError):
        MetricCollection([])
    with pytest.raises(ValueError):
        MetricCollection([AUROC(), AUROC()])
    with pytest.raises(ValueError):
        MetricCollection([FPRAtTPR(0.95), FPRAtTPR(0.95)])
    with pytest.raises(ValueError):
        MetricCollection([MetricCollection([AUROC()])])
    with pytest.raises(ValueError):
        MetricCollection([AUROC(device="cpu")])
    with pytest.raises(ValueError):
        MetricCollection([AUROC(void_label=1)])


def test_invalid_update():
    scores, labels, _ = _cat(_batches())
    collection = MetricCollection([AUROC()])
    with pytest.raises(TypeError):
        collection.update(scores, labels)
    with pytest.raises(TypeError):
        collection.update(scores=scores, labels=labels, logits=scores)
    with pytest.raises(ValueError):
        collection.update(scores=scores, labels=labels[:-1])


def test_keys():
    assert AUPR().keys == ("AUPR-OUT",)
    assert AveragePrecision().keys == ("AP-OUT",)
    assert FPRAtTPR().keys == ("FPR95TPR",)
    assert FPRAtTPR(0.95).keys == ("FPR95TPR",)
    assert FPRAtTPR(0.9).keys == ("FPR90TPR",)
    assert FPRAtTPR(0.925).keys == ("FPR92.5TPR",)
    assert AUPR("id").keys == ("AUPR-IN",)
    assert AUPR("ood").keys == ("AUPR-OUT",)
    assert AveragePrecision("id").keys == ("AP-IN",)
    assert AveragePrecision("ood").keys == ("AP-OUT",)
    assert OODMetrics().keys == ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR", "ACC")
    assert OODMetrics(fpr_at=0.9).keys[4] == "FPR90TPR"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        AUPR("out")
    with pytest.raises(ValueError):
        AveragePrecision("out")
    for tpr in (-0.1, 1.1, 95):
        with pytest.raises(ValueError):
            FPRAtTPR(tpr)
        with pytest.raises(ValueError):
            OODMetrics(fpr_at=tpr)
