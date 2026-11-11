"""
The functional metrics, compared against scikit-learn on many kinds of data.
"""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from pytorch_ood.metrics import aurra, calc_openness, calibration_error, oscr_score
from pytorch_ood.metrics import functional as F
from tests.helpers import requires

try:
    from sklearn import metrics as skm

    SKLEARN = True
except ImportError:  # pragma: no cover
    SKLEARN = False


def _reference(scores: torch.Tensor, labels: torch.Tensor, tpr: float = 0.95) -> dict:
    """sklearn values; OOD (label < 0) is the positive class."""
    s = scores.detach().double().cpu().numpy()
    ood = (labels < 0).cpu().numpy()
    precision, recall, _ = skm.precision_recall_curve(ood, s)
    precision_in, recall_in, _ = skm.precision_recall_curve(~ood, -s)
    fpr, tpr_curve, _ = skm.roc_curve(ood, s, drop_intermediate=False)
    return {
        "auroc": skm.roc_auc_score(ood, s),
        "aupr_out": skm.auc(recall, precision),
        "aupr_in": skm.auc(recall_in, precision_in),
        "ap_out": skm.average_precision_score(ood, s),
        "ap_in": skm.average_precision_score(~ood, -s),
        "fpr": fpr[np.searchsorted(tpr_curve, tpr, side="left")],
    }


def _ours(scores: torch.Tensor, labels: torch.Tensor, tpr: float = 0.95) -> dict:
    return {
        "auroc": float(F.auroc(scores, labels)),
        "aupr_out": float(F.aupr(scores, labels, positive="ood")),
        "aupr_in": float(F.aupr(scores, labels, positive="id")),
        "ap_out": float(F.average_precision(scores, labels, positive="ood")),
        "ap_in": float(F.average_precision(scores, labels, positive="id")),
        "fpr": float(F.fpr_at_tpr(scores, labels, tpr)),
    }


def _data(seed, n, ood_fraction=0.3, tie_levels=None, scale=1.0, offset=0.0, shift=1.0):
    """
    Scores of ID and OOD samples, OOD shifted by ``shift``; ``tie_levels`` rounds the scores to
    that many distinct values, which creates ties within and between the classes.
    """
    g = torch.Generator().manual_seed(seed)
    n_ood = max(1, min(n - 1, round(n * ood_fraction)))
    labels = torch.cat(
        [torch.randint(0, 10, (n - n_ood,), generator=g), -torch.ones(n_ood)]
    ).long()
    labels = labels[torch.randperm(n, generator=g)]
    scores = torch.randn(n, generator=g, dtype=torch.float64) + shift * (labels < 0).double()
    if tie_levels is not None:
        low, high = scores.min(), scores.max()
        scores = torch.round((scores - low) / (high - low) * (tie_levels - 1))
    return scores * scale + offset, labels


def _assert_matches(scores, labels, tpr=0.95):
    expected = _reference(scores, labels, tpr)
    actual = _ours(scores, labels, tpr)
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=5e-11), key


@pytest.mark.skipif(not SKLEARN, reason="scikit-learn is the reference")
class TestAgainstSklearn:
    @pytest.mark.parametrize(
        "data_seed, n", itertools.product(range(5), (2, 3, 10, 101, 1000, 20000))
    )
    def test_random_data(self, data_seed, n):
        _assert_matches(*_data(data_seed, n))

    @pytest.mark.parametrize("data_seed, levels", itertools.product(range(3), (1, 2, 3, 10, 100)))
    def test_ties(self, data_seed, levels):
        _assert_matches(*_data(data_seed, 2000, tie_levels=levels))

    @pytest.mark.parametrize(
        "scale, offset", itertools.product((1e-6, 1.0, 1e3, 1e6), (0.0, 1e4, -1e4, 1e8))
    )
    def test_magnitudes(self, scale, offset):
        # torchmetrics applied a sigmoid here, which saturates and turned the scores into ties
        scores, labels = _data(0, 2000, tie_levels=200)
        _assert_matches(scores * scale + offset, labels)

    @pytest.mark.parametrize("fraction", (0.001, 0.01, 0.5, 0.99, 0.999))
    def test_imbalance(self, fraction):
        _assert_matches(*_data(1, 5000, ood_fraction=fraction))

    def test_perfect_and_reversed(self):
        labels = torch.cat([torch.zeros(50), -torch.ones(50)]).long()
        scores = torch.arange(100.0)
        _assert_matches(scores, labels)
        _assert_matches(-scores, labels)
        assert float(F.auroc(scores, labels)) == 1.0
        assert float(F.auroc(-scores, labels)) == 0.0

    def test_dtypes(self):
        scores, labels = _data(2, 3000)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            # the reference gets the same (rounded) values
            _assert_matches(scores.to(dtype), labels)
        _assert_matches(*_data(2, 3000, tie_levels=50, scale=1.0))
        int_scores = _data(3, 3000, tie_levels=50)[0].long()
        _assert_matches(int_scores, labels)
        _assert_matches((int_scores > 25), labels)

    @pytest.mark.parametrize(
        "dtype", (torch.int8, torch.int16, torch.int32, torch.float32, torch.float64)
    )
    def test_label_dtypes(self, dtype):
        scores, labels = _data(4, 500)
        expected = _ours(scores, labels)
        assert _ours(scores, labels.to(dtype)) == expected

    def test_shapes(self):
        # inputs of any shape are flattened
        scores, labels = _data(5, 600)
        expected = _ours(scores, labels)
        assert _ours(scores.view(2, 3, 100), labels.view(2, 3, 100)) == expected

    @pytest.mark.parametrize("tpr", (0.0, 0.05, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0))
    def test_tpr_values(self, tpr):
        scores, labels = _data(6, 1000, tie_levels=30)
        _assert_matches(scores, labels, tpr=tpr)

    @requires(cuda=1)
    def test_cuda(self):
        scores, labels = _data(7, 50000, tie_levels=1000)
        expected = _ours(scores, labels)
        actual = _ours(scores.cuda(), labels.cuda())
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], abs=5e-13), key

    def test_curves_match_sklearn(self):
        scores, labels = _data(8, 500, tie_levels=40)
        ood = (labels < 0).numpy()
        fpr, tpr, thresholds = F.roc_curve(scores, labels)
        sk_fpr, sk_tpr, sk_thresholds = skm.roc_curve(ood, scores.numpy(), drop_intermediate=False)
        np.testing.assert_allclose(fpr.numpy(), sk_fpr)
        np.testing.assert_allclose(tpr.numpy(), sk_tpr)
        # sklearn prepends an infinite threshold for the first point
        np.testing.assert_allclose(thresholds.numpy(), sk_thresholds[1:])

        precision, recall, thresholds = F.pr_curve(scores, labels)
        sk_precision, sk_recall, sk_thresholds = skm.precision_recall_curve(ood, scores.numpy())
        np.testing.assert_allclose(precision.numpy(), sk_precision)
        np.testing.assert_allclose(recall.numpy(), sk_recall)
        np.testing.assert_allclose(thresholds.numpy(), sk_thresholds)

        precision, recall, thresholds = F.pr_curve(scores, labels, positive="id")
        sk_precision, sk_recall, sk_thresholds = skm.precision_recall_curve(~ood, -scores.numpy())
        np.testing.assert_allclose(precision.numpy(), sk_precision)
        np.testing.assert_allclose(recall.numpy(), sk_recall)
        np.testing.assert_allclose(-thresholds.numpy(), sk_thresholds)


class TestEdgeCases:
    @pytest.mark.parametrize(
        "fn",
        (
            F.auroc,
            F.aupr,
            F.average_precision,
            F.fpr_at_tpr,
            F.roc_curve,
            F.pr_curve,
            F.autc,
        ),
    )
    def test_nan_raises(self, fn):
        scores = torch.tensor([0.0, float("nan"), 1.0, 2.0])
        labels = torch.tensor([0, 0, -1, -1])
        with pytest.raises(ValueError):
            fn(scores, labels)

    def test_infinite_scores(self):
        labels = torch.tensor([0, 0, 0, -1, -1, -1])
        scores = torch.tensor([float("-inf"), 0.0, 5.0, 3.0, 10.0, float("inf")])
        # the same order with finite scores
        finite = torch.tensor([-100.0, 0.0, 5.0, 3.0, 10.0, 100.0])
        assert _ours(scores, labels) == _ours(finite, labels)
        # infinite scores of both classes tie
        both = torch.tensor([float("inf"), 0.0, 0.0, float("inf"), 1.0, 1.0])
        both_finite = torch.tensor([100.0, 0.0, 0.0, 100.0, 1.0, 1.0])
        assert _ours(both, labels) == _ours(both_finite, labels)

    def test_negative_zero_ties_with_zero(self):
        assert float(F.auroc(torch.tensor([0.0, -0.0]), torch.tensor([0, -1]))) == 0.5
        assert float(F.auroc(torch.tensor([-0.0, 0.0]), torch.tensor([0, -1]))) == 0.5

    @pytest.mark.parametrize("labels", (torch.zeros(5).long(), -torch.ones(5).long()))
    @pytest.mark.parametrize("fn", (F.auroc, F.aupr, F.average_precision, F.fpr_at_tpr, F.autc))
    def test_one_class_raises(self, labels, fn):
        with pytest.raises(ValueError):
            fn(torch.arange(5.0), labels)

    @pytest.mark.parametrize("fn", (F.auroc, F.aupr, F.average_precision))
    def test_shape_mismatch_raises(self, fn):
        with pytest.raises(ValueError):
            fn(torch.zeros(4), torch.zeros(5))
        with pytest.raises(ValueError):
            fn(torch.zeros(2, 3), torch.zeros(3, 2))

    def test_invalid_arguments(self):
        scores, labels = torch.arange(4.0), torch.tensor([0, 0, -1, -1])
        with pytest.raises(ValueError):
            F.aupr(scores, labels, positive="out")
        with pytest.raises(ValueError):
            F.average_precision(scores, labels, positive="out")
        with pytest.raises(ValueError):
            F.pr_curve(scores, labels, positive="in")
        for tpr in (-0.1, 1.5, 95):
            with pytest.raises(ValueError):
                F.fpr_at_tpr(scores, labels, tpr)

    def test_does_not_modify_inputs(self):
        scores, labels = _data(0, 100)
        before = scores.clone(), labels.clone()
        _ours(scores, labels)
        F.autc(scores, labels)
        assert torch.equal(scores, before[0])
        assert torch.equal(labels, before[1])

    def test_requires_grad(self):
        scores = torch.randn(20, requires_grad=True)
        labels = torch.tensor([0] * 10 + [-1] * 10)
        assert not F.auroc(scores, labels).requires_grad

    def _check_float32_counts(self, device: str) -> None:
        # float32 represents integers above 2^24 only approximately (odd ones not at all); the
        # counts must stay exact. CUDA accumulates float32 sums in float32 (the CPU in float64),
        # so both are checked.
        n_low = 2**24
        scores = torch.cat(
            [torch.zeros(n_low), torch.full((2001,), 2.0), torch.ones(1001), -torch.ones(1000)]
        )
        labels = torch.cat([torch.zeros(n_low + 2001), -torch.ones(2001)]).long()
        n_id, n_ood = n_low + 2001, 2001
        # OOD at 1 are above the n_low ID at 0, OOD at -1 are below all
        expected = Fraction(1001 * n_low, n_id * n_ood)
        s, y = scores.to(device), labels.to(device)
        counts = F._Counts(s, y < 0)
        assert counts.fps.tolist() == [2001, 2001, n_id, n_id]
        assert counts.tps.tolist() == [0, 1001, 1001, n_ood]
        assert float(F.auroc(s, y)) == pytest.approx(float(expected), abs=5e-13)
        assert float(F.fpr_at_tpr(s, y, 0.5)) == 2001 / n_id
        # thresholds 1 (1001 of 2001 OOD, 2001 ID above) and -1 (all samples) raise
        # the recall of OOD; 0 (n_low ID) and 2 (the remaining ID) that of ID
        ap_out = Fraction(1001, 2001) * Fraction(1001, 3002) + Fraction(1000, 2001) * Fraction(
            n_ood, n_id + n_ood
        )
        ap_in = Fraction(n_low, n_id) * Fraction(n_low, n_low + 1000) + Fraction(
            2001, n_id
        ) * Fraction(n_id, n_id + n_ood)
        assert float(F.average_precision(s, y, "ood")) == pytest.approx(float(ap_out), abs=5e-13)
        assert float(F.average_precision(s, y, "id")) == pytest.approx(float(ap_in), abs=5e-13)
        # the same with the classes swapped, so that the OOD counts are large
        counts = F._Counts(-s, y >= 0)
        assert counts.tps.tolist() == [0, n_low, n_low, n_id]
        assert counts.fps.tolist() == [1000, 1000, 2001, 2001]

    def test_more_samples_than_float32_counts(self):
        self._check_float32_counts("cpu")

    @requires(cuda=1)
    def test_more_samples_than_float32_counts_on_cuda(self):
        self._check_float32_counts("cuda")


def _exact_pr_areas(scores, positive):
    """
    Average precision and trapezoidal AUPR as exact fractions, from the definitions: one point
    of the curve per unique score, counted directly, without sorting or cumulative sums.
    """
    scores, positive = scores.tolist(), positive.tolist()
    n_pos = sum(positive)
    points = [(Fraction(0), Fraction(1))]  # (recall, precision), by decreasing threshold
    for t in sorted(set(scores), reverse=True):
        tp = sum(1 for s, p in zip(scores, positive) if s >= t and p)
        fp = sum(1 for s, p in zip(scores, positive) if s >= t and not p)
        points.append((Fraction(tp, n_pos), Fraction(tp, tp + fp)))
    ap = sum((r - r_prev) * p for (r_prev, _), (r, p) in zip(points, points[1:]))
    trapezoid = sum(
        (r - r_prev) * (p + p_prev) / 2 for (r_prev, p_prev), (r, p) in zip(points, points[1:])
    )
    return ap, trapezoid


class TestAveragePrecision:
    @pytest.mark.parametrize("data_seed, levels", itertools.product(range(3), (2, 5, 30, None)))
    def test_matches_definition(self, data_seed, levels):
        scores, labels = _data(data_seed, 300, tie_levels=levels)
        for positive, mask, s in (
            ("ood", labels < 0, scores),
            ("id", labels >= 0, -scores),
        ):
            ap, trapezoid = _exact_pr_areas(s, mask)
            assert float(F.average_precision(scores, labels, positive)) == pytest.approx(
                float(ap), abs=5e-13
            )
            assert float(F.aupr(scores, labels, positive)) == pytest.approx(
                float(trapezoid), abs=5e-13
            )

    def test_results_are_detached_scalars(self):
        scores = torch.randn(20, requires_grad=True)
        labels = torch.tensor([0] * 10 + [-1] * 10)
        precision = torch.tensor([0.4, 0.75, 1.0], requires_grad=True)
        recall = torch.tensor([1.0, 0.75, 0.0], requires_grad=True)
        for value in (
            F.average_precision(scores, labels),
            F.average_precision_from_curve(precision, recall),
            F.aupr_from_curve(precision, recall),
        ):
            assert not value.requires_grad
            assert value.dim() == 0
            assert value.dtype == torch.float64

    def test_from_curve_equals_functional(self):
        for seed, levels in itertools.product(range(3), (1, 2, 10, None)):
            scores, labels = _data(seed, 1000, tie_levels=levels)
            for positive in ("ood", "id"):
                precision, recall, _ = F.pr_curve(scores, labels, positive)
                assert float(F.average_precision_from_curve(precision, recall)) == float(
                    F.average_precision(scores, labels, positive)
                )
                assert float(F.aupr_from_curve(precision, recall)) == float(
                    F.aupr(scores, labels, positive)
                )

    def test_from_curve_of_histogram(self):
        # a curve from two bins: 30 OOD and 10 ID samples in the upper bin, 10 OOD and 50 ID in
        # the lower bin, plus the final point at recall 0
        precision = torch.tensor([40 / 100, 30 / 40, 1.0], dtype=torch.float32)
        recall = torch.tensor([1.0, 30 / 40, 0.0], dtype=torch.float32)
        ap = F.average_precision_from_curve(precision, recall)
        assert ap.dtype == torch.float64
        assert float(ap) == pytest.approx(0.75 * 0.75 + 0.25 * 0.4, abs=5e-8)
        assert float(F.aupr_from_curve(precision, recall)) == pytest.approx(
            0.75 * (1 + 0.75) / 2 + 0.25 * (0.75 + 0.4) / 2, abs=5e-8
        )
        # the same as from the samples
        scores = torch.cat([torch.ones(40), torch.zeros(60)])
        labels = torch.cat([-torch.ones(30), torch.zeros(10), -torch.ones(10), torch.zeros(50)])
        assert float(F.average_precision(scores, labels)) == pytest.approx(float(ap), abs=5e-8)
        # a plateau of recall (a threshold that adds only negatives) adds no step; the step
        # below it uses the precision of its last point, the highest threshold with that recall
        precision = torch.tensor([0.4, 0.3, 0.75, 1.0])
        recall = torch.tensor([1.0, 0.75, 0.75, 0.0])
        assert float(F.average_precision_from_curve(precision, recall)) == pytest.approx(
            0.25 * 0.4 + 0.75 * 0.75, abs=5e-8
        )

    def test_from_curve_rejects_invalid_curves(self):
        precision = torch.tensor([0.4, 0.75, 1.0])
        recall = torch.tensor([1.0, 0.75, 0.0])
        p4 = torch.tensor([0.4, 0.5, 0.75, 1.0])
        # each case violates only the check named by the expected message
        cases = {
            "length mismatch": (precision, recall[1:], "same length"),
            "2-D": (precision.view(1, 3), recall.view(1, 3), "one-dimensional"),
            "0-D": (precision[0], recall[0], "one-dimensional"),
            "single point": (precision[-1:], recall[-1:], "at least two points"),
            "precision above 1": (torch.tensor([0.4, 1.5, 1.0]), recall, "precision contains"),
            "negative precision": (torch.tensor([-0.4, 0.75, 1.0]), recall, "precision contains"),
            "NaN precision": (torch.tensor([0.4, float("nan"), 1.0]), recall, "0 / 0"),
            "recall above 1": (precision, torch.tensor([1.0, 1.5, 0.0]), "recall contains"),
            "recall below 0": (p4, torch.tensor([1.0, 0.5, -0.5, 0.0]), "recall contains"),
            "NaN recall": (precision, torch.tensor([1.0, float("nan"), 0.0]), "recall contains"),
            "increasing recall": (precision, torch.tensor([0.5, 1.0, 0.0]), "non-increasing"),
            "unsorted recall": (p4, torch.tensor([1.0, 0.5, 0.75, 0.0]), "non-increasing"),
            "slightly increasing recall": (
                p4,
                torch.tensor([1.0, 0.5, 0.5 + 1e-9, 0.0], dtype=torch.float64),
                "non-increasing",
            ),
            "not starting at recall 1": (precision, torch.tensor([0.9, 0.5, 0.0]), "start at"),
            "not ending at recall 0": (
                precision,
                torch.tensor([1.0, 0.75, 0.25]),
                "end at recall",
            ),
        }
        for fn in (F.average_precision_from_curve, F.aupr_from_curve):
            for name, (p, r, message) in cases.items():
                with pytest.raises(ValueError, match=message):
                    fn(p, r)

    def test_from_curve_accepts_valid_edge_cases(self):
        precision = torch.tensor([0.4, 0.5, 0.75, 1.0])
        cases = {
            # a final -0.0 equals 0
            "ends at -0.0": (
                torch.tensor([1.0, 0.5, 0.25, -0.0]),
                0.5 * 0.4 + 0.25 * 0.5 + 0.25 * 0.75,
            ),
            # a threshold that adds only negatives at full recall
            "start plateau": (torch.tensor([1.0, 1.0, 0.5, 0.0]), 0.5 * 0.5 + 0.5 * 0.75),
        }
        for name, (recall, expected) in cases.items():
            ap = F.average_precision_from_curve(precision, recall)
            assert float(ap) == pytest.approx(expected, abs=5e-8)
            F.aupr_from_curve(precision, recall)

    def test_from_curve_final_precision(self):
        recall = torch.tensor([1.0, 0.75, 0.0])
        ones = torch.tensor([0.4, 0.75, 1.0])
        other = torch.tensor([0.4, 0.75, 0.3])
        # AP does not use the precision at recall 0
        assert float(F.average_precision_from_curve(other, recall)) == float(
            F.average_precision_from_curve(ones, recall)
        )
        # the trapezoidal rule does, so it requires the end point of pr_curve
        with pytest.raises(ValueError, match="end at precision 1"):
            F.aupr_from_curve(other, recall)

    def test_from_curve_dtypes(self):
        precision = torch.tensor([0.5, 0.75, 1.0])
        recall = torch.tensor([1.0, 0.5, 0.0])
        expected = 0.5 * 0.5 + 0.5 * 0.75
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            ap = F.average_precision_from_curve(precision.to(dtype), recall.to(dtype))
            assert ap.dtype == torch.float64
            assert float(ap) == expected
            aupr = F.aupr_from_curve(precision.to(dtype), recall.to(dtype))
            assert aupr.dtype == torch.float64
        ap = F.average_precision_from_curve(torch.tensor([0.5, 1.0]), torch.tensor([1, 0]))
        assert ap.dtype == torch.float64
        assert float(ap) == 0.5

    @pytest.mark.parametrize("fn", (F.average_precision_from_curve, F.aupr_from_curve))
    def test_from_curve_of_long_float32_curve(self, fn):
        g = torch.Generator().manual_seed(0)
        n = 2**20
        labels = -(torch.rand(n, generator=g) < 0.3).long()
        scores = torch.randn(n, generator=g, dtype=torch.float64) + (labels < 0).double()
        precision, recall, _ = F.pr_curve(scores, labels)
        assert precision.numel() > n // 2
        expected = float(F.average_precision_from_curve(precision, recall))
        assert expected == float(F.average_precision(scores, labels))
        full = float(fn(precision, recall))
        single = float(fn(precision.float(), recall.float()))
        assert single == pytest.approx(full, abs=1e-6)


@requires(cuda=1)
def test_labels_on_other_device():
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(-1, 3, (200,), generator=g)
    scores = torch.randn(200, generator=g) + (labels < 0).float()
    for fn in (
        F.auroc,
        F.autc,
        lambda s, y: F.fpr_at_tpr(s, y, 0.95),
        lambda s, y: F.aupr(s, y, "id"),
        lambda s, y: F.average_precision(s, y, "ood"),
        lambda s, y: F.average_precision(s, y, "id"),
    ):
        assert float(fn(scores.cuda(), labels)) == pytest.approx(
            float(fn(scores, labels)), abs=5e-8
        )


@requires(cuda=1)
def test_result_stays_on_device():
    scores, labels = _data(0, 500, tie_levels=20)
    scores, labels = scores.cuda(), labels.cuda()
    precision, recall, _ = F.pr_curve(scores, labels)
    assert precision.device.type == "cuda"
    for value in (
        F.aupr(scores, labels),
        F.average_precision(scores, labels),
        F.average_precision(scores, labels, positive="id"),
        F.aupr_from_curve(precision, recall),
        F.average_precision_from_curve(precision, recall),
        F.average_precision_from_curve(precision, recall.cpu()),
    ):
        assert value.device.type == "cuda"
        assert value.dtype == torch.float64


@pytest.mark.skipif(not SKLEARN, reason="scikit-learn is the reference")
def test_matches_threshold_curve():
    # the closed form equals the integral of the threshold curves over [0, 1]
    g = torch.Generator().manual_seed(3)
    scores = torch.randn(5000, generator=g, dtype=torch.float64)
    labels = -(torch.rand(5000, generator=g) > 0.5).long()
    s = ((scores - scores.min()) / (scores.max() - scores.min())).numpy()
    ood = (labels < 0).numpy()
    tau = np.linspace(0, 1, 20001)
    fpr = np.array([(s[~ood] >= t).mean() for t in tau])
    fnr = np.array([(s[ood] < t).mean() for t in tau])
    area = lambda y: float(((y[1:] + y[:-1]) / 2 * np.diff(tau)).sum())  # noqa: E731
    reference = (area(fpr) + area(fnr)) / 2
    assert float(F.autc(scores, labels)) == pytest.approx(reference, abs=5e-4)


def test_constant_scores_raise():
    with pytest.raises(ValueError):
        F.autc(torch.ones(10), torch.tensor([0] * 5 + [-1] * 5))


def test_infinite_scores_raise():
    with pytest.raises(ValueError):
        F.autc(torch.tensor([0.0, 1.0, float("inf")]), torch.tensor([0, -1, -1]))


def test_large_class_indices():
    labels = torch.tensor([70000, 1000, 5])
    assert float(F.accuracy(labels.clone(), labels)) == 1.0


def test_no_id_raises():
    with pytest.raises(ValueError):
        F.accuracy(torch.zeros(3).long(), -torch.ones(3).long())


@pytest.mark.skipif(not SKLEARN, reason="scikit-learn is the reference")
def test_matches_sklearn():
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(-1, 10, (1000,), generator=g)
    predictions = torch.randint(0, 10, (1000,), generator=g)
    known = labels >= 0
    expected = skm.accuracy_score(labels[known].numpy(), predictions[known].numpy())
    assert float(F.accuracy(predictions, labels)) == pytest.approx(expected, abs=5e-13)


class TestMovedFunctions:
    @staticmethod
    def _three_bins():
        # bins of 100 samples: confidence 0.2 / 0.5 / 0.9, accuracy 0.2 / 0.0 / 0.6,
        # i.e. calibration gaps 0.0 / 0.5 / 0.3
        conf = torch.cat([torch.full((100,), c) for c in (0.2, 0.5, 0.9)])
        correct = torch.zeros(300)
        correct[:20] = 1
        correct[200:260] = 1
        return conf, correct

    def test_calibration_error_matches_hand_computed(self):
        conf, correct = self._three_bins()
        expected = {
            "1": (0.0 + 0.5 + 0.3) / 3,
            "2": math.sqrt((0.0 + 0.25 + 0.09) / 3),
            "infty": 0.5,
        }
        for p, value in expected.items():
            assert calibration_error(conf, correct, p=p, beta=100) == pytest.approx(
                value, abs=5e-8
            )

    def test_calibration_error_includes_last_bin(self):
        # only the highest-confidence bin is miscalibrated
        conf = torch.cat([torch.full((100,), 0.5), torch.full((100,), 1.0)])
        correct = torch.cat([torch.ones(50), torch.zeros(150)])
        assert calibration_error(conf, correct, p="1", beta=100) == pytest.approx(0.5, abs=5e-8)

    def test_calibration_error_last_bin_takes_remainder(self):
        # 250 samples: bins of 100 and 150 samples, gaps 0.0 and 0.4
        conf = torch.cat([torch.full((100,), 0.5), torch.full((150,), 0.9)])
        correct = torch.cat([torch.ones(50), torch.zeros(50), torch.ones(75), torch.zeros(75)])
        assert calibration_error(conf, correct, p="1", beta=100) == pytest.approx(
            150 / 250 * 0.4, abs=5e-8
        )

    def test_calibration_error_with_fewer_samples_than_bin_size(self):
        # a single bin with all samples
        conf = torch.tensor([0.9, 0.9, 0.6, 0.6])
        correct = torch.tensor([1, 0, 0, 0])
        assert calibration_error(conf, correct, p="1", beta=100) == pytest.approx(
            0.75 - 0.25, abs=5e-8
        )

    @pytest.mark.parametrize("p", ("1", "2", "infty"))
    def test_calibration_error_of_calibrated_predictions_is_zero(self, p):
        # one bin per confidence level, with matching accuracy
        conf = torch.cat([torch.full((100,), 0.25), torch.full((100,), 0.75)])
        correct = torch.cat([torch.ones(25), torch.zeros(75), torch.ones(75), torch.zeros(25)])
        assert calibration_error(conf, correct, p=p, beta=100) == pytest.approx(0.0, abs=5e-8)

    def test_calibration_error_is_permutation_invariant(self):
        conf, correct = self._three_bins()
        perm = torch.randperm(300, generator=torch.Generator().manual_seed(0))
        assert calibration_error(conf[perm], correct[perm]) == pytest.approx(
            calibration_error(conf, correct), abs=5e-8
        )

    def test_calibration_error_rejects_invalid_input(self):
        conf, correct = self._three_bins()
        cases = {
            "unknown norm": lambda: calibration_error(conf, correct, p="3"),
            "bin size": lambda: calibration_error(conf, correct, beta=0),
            "shape": lambda: calibration_error(conf, correct[:-1]),
            "empty": lambda: calibration_error(conf[:0], correct[:0]),
            "NaN": lambda: calibration_error(torch.full((4,), float("nan")), correct[:4]),
        }
        for name, call in cases.items():
            with pytest.raises(ValueError):
                call()

    @requires(cuda=1)
    def test_calibration_error_on_cuda(self):
        conf, correct = self._three_bins()
        assert calibration_error(conf.cuda(), correct.cuda()) == pytest.approx(
            calibration_error(conf, correct), abs=5e-8
        )

    def test_aurra(self):
        assert aurra(torch.tensor([0.9, 0.1]), torch.tensor([1, 0])) == 0.75
        # answered in the order 0.9, 0.5, 0.1: accuracies 1, 1/2, 2/3
        assert aurra(torch.tensor([0.1, 0.9, 0.5]), torch.tensor([1, 1, 0])) == pytest.approx(
            (1 + 1 / 2 + 2 / 3) / 3, abs=5e-8
        )

    def test_aurra_of_any_shape(self):
        confidence = torch.rand(4, 5, generator=torch.Generator().manual_seed(0))
        correct = (confidence > 0.3).long()
        assert aurra(confidence, correct) == aurra(confidence.flatten(), correct.flatten())

    def test_aurra_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            aurra(torch.rand(4), torch.ones(3))
        with pytest.raises(ValueError):
            aurra(torch.zeros(0), torch.zeros(0))
        with pytest.raises(ValueError):
            aurra(torch.tensor([0.5, float("nan")]), torch.ones(2))

    @requires(cuda=1)
    def test_aurra_on_cuda(self):
        confidence = torch.rand(100, generator=torch.Generator().manual_seed(0))
        correct = (confidence > 0.3).long()
        assert aurra(confidence.cuda(), correct.cuda()) == pytest.approx(
            aurra(confidence, correct), abs=5e-8
        )

    def test_oscr_of_any_shape(self):
        g = torch.Generator().manual_seed(0)
        labels = torch.randint(-1, 3, (4, 6), generator=g)
        scores = torch.randn(4, 6, generator=g)
        predictions = torch.randint(0, 3, (4, 6), generator=g)
        assert oscr_score(scores, predictions, labels) == oscr_score(
            scores.flatten(), predictions.flatten(), labels.flatten()
        )

    def test_oscr_rejects_invalid_input(self):
        labels = torch.tensor([0, 1, -1, -1])
        predictions = torch.tensor([0, 1, 0, 0])
        with pytest.raises(ValueError):
            oscr_score(torch.rand(4), predictions[:3], labels)
        with pytest.raises(ValueError):
            oscr_score(torch.rand(3), predictions, labels)
        with pytest.raises(ValueError):
            oscr_score(torch.tensor([0.0, 1.0, float("nan"), 2.0]), predictions, labels)

    @requires(cuda=1)
    def test_oscr_on_cuda(self):
        g = torch.Generator().manual_seed(0)
        labels = torch.randint(-1, 3, (100,), generator=g)
        scores = torch.randn(100, generator=g)
        predictions = torch.randint(0, 3, (100,), generator=g)
        assert oscr_score(scores.cuda(), predictions.cuda(), labels) == pytest.approx(
            oscr_score(scores, predictions, labels), abs=5e-8
        )

    def test_openness(self):
        assert calc_openness(n_train=6, n_test=10, n_target=6) == pytest.approx(
            1 - float(np.sqrt(12 / 16)), abs=5e-8
        )

    def test_openness_of_closed_set_is_zero(self):
        assert calc_openness(n_train=6, n_test=6, n_target=6) == 0
