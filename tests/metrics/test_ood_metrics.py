"""
OODMetrics: the fixed collection of the commonly reported metrics.
"""

import math
import warnings

import pytest
import torch

from pytorch_ood.metrics import AUTC, OODMetrics
from pytorch_ood.metrics import functional as F


def _separated(n=10):
    scores = torch.zeros(n)
    labels = torch.zeros(n).long()
    scores[n // 2 :] = 1
    labels[n // 2 :] = -1
    return scores, labels


def test_does_not_include_average_precision():
    scores, labels = _separated()
    result = OODMetrics().update(scores, labels, labels.clamp(min=0)).compute()
    assert tuple(result) == ("AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR", "ACC")


def test_values_match_functional():
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


def test_no_accuracy_without_predictions():
    scores, labels = _separated()
    assert "ACC" not in OODMetrics().update(scores, labels).compute()


def test_predictions_in_some_updates_only_raise():
    scores, labels = _separated()
    metrics = OODMetrics().update(scores, labels, labels.clamp(min=0))
    with pytest.raises(ValueError):
        metrics.update(scores, labels)
    metrics = OODMetrics().update(scores, labels)
    with pytest.raises(ValueError):
        metrics.update(scores, labels, labels.clamp(min=0))


def test_predictions_after_reset():
    scores, labels = _separated()
    metrics = OODMetrics().update(scores, labels, labels.clamp(min=0))
    metrics.reset()
    assert "ACC" not in metrics.update(scores, labels).compute()


def test_accuracy_without_id_samples_raises():
    scores, labels = _separated()
    metrics = OODMetrics(void_label=0).update(scores, labels, labels.clamp(min=0))
    with pytest.raises(ValueError):
        metrics.compute()


def test_shape_mismatch():
    scores, labels = _separated()
    with pytest.raises(ValueError):
        OODMetrics().update(scores, labels, torch.zeros(5).long())
    with pytest.raises(ValueError):
        OODMetrics().update(scores, labels[:5])


def test_only_id_or_only_ood_raises():
    for labels in (torch.zeros(10).long(), -torch.ones(10).long()):
        metrics = OODMetrics().update(torch.arange(10.0), labels)
        with pytest.raises(ValueError):
            metrics.compute()


def test_inputs_of_any_shape():
    scores, labels = _separated(16)
    expected = OODMetrics().update(scores, labels).compute()
    assert OODMetrics().update(scores.view(2, 2, 4), labels.view(2, 2, 4)).compute() == expected


def test_autc_is_nan_for_constant_scores():
    labels = torch.tensor([0, 0, -1, -1])
    with pytest.warns(UserWarning):
        result = OODMetrics().update(torch.ones(4), labels).compute()
    assert math.isnan(result["AUTC"])
    assert result["AUROC"] == 0.5


def test_autc_is_nan_for_infinite_scores():
    labels = torch.tensor([0, 0, 0, -1, -1, -1])
    scores = torch.tensor([-float("inf"), 0.0, 1.0, 1.0, 2.0, float("inf")])
    with pytest.warns(UserWarning):
        result = OODMetrics().update(scores, labels).compute()
    assert math.isnan(result["AUTC"])
    # infinite scores rank like any other: same as finite scores in the same order
    expected = OODMetrics().update(torch.tensor([-9.0, 0, 1, 1, 2, 9]), labels).compute()
    for key in ("AUROC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"):
        assert result[key] == expected[key], key


def test_autc_raises_for_invalid_data():
    # only constant and infinite scores give NaN; invalid data raises as for the others
    labels = torch.tensor([0, 0, -1, -1])
    cases = {
        "NaN": (torch.tensor([0.0, float("nan"), 1.0, 2.0]), labels),
        "only ID": (torch.ones(4), torch.zeros(4).long()),
        "only OOD": (torch.arange(4.0), -torch.ones(4).long()),
    }
    for name, (scores, y) in cases.items():
        metric = AUTC().update(scores, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                metric.compute()


def test_no_warning_for_valid_scores():
    scores, labels = _separated()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        OODMetrics().update(scores, labels).compute()
