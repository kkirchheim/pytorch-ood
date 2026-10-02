"""
The functional metrics, compared against scikit-learn on many kinds of data.
"""

import itertools
import unittest
from fractions import Fraction

import numpy as np
import torch

from src.pytorch_ood.metrics import aurra, calc_openness, calibration_error
from src.pytorch_ood.metrics import functional as F

try:
    from sklearn import metrics as skm

    SKLEARN = True
except ImportError:  # pragma: no cover
    SKLEARN = False

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


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
        "fpr": fpr[np.searchsorted(tpr_curve, tpr, side="left")],
    }


def _ours(scores: torch.Tensor, labels: torch.Tensor, tpr: float = 0.95) -> dict:
    return {
        "auroc": float(F.auroc(scores, labels)),
        "aupr_out": float(F.aupr(scores, labels, positive="ood")),
        "aupr_in": float(F.aupr(scores, labels, positive="id")),
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


@unittest.skipUnless(SKLEARN, "scikit-learn is the reference")
class TestAgainstSklearn(unittest.TestCase):
    def assertMatches(self, scores, labels, tpr=0.95, places=10):
        expected = _reference(scores, labels, tpr)
        actual = _ours(scores, labels, tpr)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=places, msg=key)

    def test_random_data(self):
        for seed, n in itertools.product(range(5), (2, 3, 10, 101, 1000, 20000)):
            with self.subTest(seed=seed, n=n):
                self.assertMatches(*_data(seed, n))

    def test_ties(self):
        for seed, levels in itertools.product(range(3), (1, 2, 3, 10, 100)):
            with self.subTest(seed=seed, levels=levels):
                self.assertMatches(*_data(seed, 2000, tie_levels=levels))

    def test_all_scores_equal(self):
        scores = torch.full((100,), 3.0)
        labels = torch.cat([torch.zeros(70), -torch.ones(30)]).long()
        self.assertMatches(scores, labels)
        self.assertEqual(float(F.auroc(scores, labels)), 0.5)

    def test_magnitudes(self):
        # torchmetrics applied a sigmoid here, which saturates and turned the scores into ties
        for scale, offset in itertools.product((1e-6, 1.0, 1e3, 1e6), (0.0, 1e4, -1e4, 1e8)):
            with self.subTest(scale=scale, offset=offset):
                scores, labels = _data(0, 2000, tie_levels=200)
                self.assertMatches(scores * scale + offset, labels)

    def test_separated_large_scores(self):
        for offset in (0.0, 1000.0, -1000.0):
            with self.subTest(offset=offset):
                scores = torch.tensor([20.0, 30.0, 40.0, 50.0]) + offset
                labels = torch.tensor([0, 0, -1, -1])
                result = _ours(scores, labels)
                self.assertEqual(
                    result, {"auroc": 1.0, "aupr_out": 1.0, "aupr_in": 1.0, "fpr": 0.0}
                )

    def test_imbalance(self):
        for fraction in (0.001, 0.01, 0.5, 0.99, 0.999):
            with self.subTest(ood_fraction=fraction):
                self.assertMatches(*_data(1, 5000, ood_fraction=fraction))

    def test_single_sample_per_class(self):
        for scores in ([0.0, 1.0], [1.0, 0.0], [1.0, 1.0]):
            with self.subTest(scores=scores):
                self.assertMatches(torch.tensor(scores), torch.tensor([0, -1]))

    def test_perfect_and_reversed(self):
        labels = torch.cat([torch.zeros(50), -torch.ones(50)]).long()
        scores = torch.arange(100.0)
        self.assertMatches(scores, labels)
        self.assertMatches(-scores, labels)
        self.assertEqual(float(F.auroc(scores, labels)), 1.0)
        self.assertEqual(float(F.auroc(-scores, labels)), 0.0)

    def test_dtypes(self):
        scores, labels = _data(2, 3000)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # the reference gets the same (rounded) values
                self.assertMatches(scores.to(dtype), labels)
        self.assertMatches(*_data(2, 3000, tie_levels=50, scale=1.0))
        int_scores = _data(3, 3000, tie_levels=50)[0].long()
        self.assertMatches(int_scores, labels)
        self.assertMatches((int_scores > 25), labels)

    def test_label_dtypes(self):
        scores, labels = _data(4, 500)
        expected = _ours(scores, labels)
        for dtype in (torch.int8, torch.int16, torch.int32, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.assertEqual(_ours(scores, labels.to(dtype)), expected)

    def test_shapes(self):
        # inputs of any shape are flattened
        scores, labels = _data(5, 600)
        expected = _ours(scores, labels)
        self.assertEqual(_ours(scores.view(2, 3, 100), labels.view(2, 3, 100)), expected)

    def test_tpr_values(self):
        scores, labels = _data(6, 1000, tie_levels=30)
        for tpr in (0.0, 0.05, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
            with self.subTest(tpr=tpr):
                self.assertMatches(scores, labels, tpr=tpr)

    @unittest.skipUnless(CUDA, "requires CUDA")
    def test_cuda(self):
        scores, labels = _data(7, 50000, tie_levels=1000)
        expected = _ours(scores, labels)
        actual = _ours(scores.cuda(), labels.cuda())
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=12, msg=key)

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


class TestEdgeCases(unittest.TestCase):
    def test_nan_raises(self):
        scores = torch.tensor([0.0, float("nan"), 1.0, 2.0])
        labels = torch.tensor([0, 0, -1, -1])
        for fn in (F.auroc, F.aupr, F.fpr_at_tpr, F.roc_curve, F.pr_curve, F.autc):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ValueError):
                    fn(scores, labels)

    def test_infinite_scores(self):
        labels = torch.tensor([0, 0, 0, -1, -1, -1])
        scores = torch.tensor([float("-inf"), 0.0, 5.0, 3.0, 10.0, float("inf")])
        # the same order with finite scores
        finite = torch.tensor([-100.0, 0.0, 5.0, 3.0, 10.0, 100.0])
        self.assertEqual(_ours(scores, labels), _ours(finite, labels))
        # infinite scores of both classes tie
        both = torch.tensor([float("inf"), 0.0, 0.0, float("inf"), 1.0, 1.0])
        both_finite = torch.tensor([100.0, 0.0, 0.0, 100.0, 1.0, 1.0])
        self.assertEqual(_ours(both, labels), _ours(both_finite, labels))

    def test_infinite_scores_tie(self):
        labels = torch.tensor([0, -1])
        self.assertEqual(float(F.auroc(torch.tensor([float("inf"), float("inf")]), labels)), 0.5)

    def test_negative_zero_ties_with_zero(self):
        self.assertEqual(float(F.auroc(torch.tensor([0.0, -0.0]), torch.tensor([0, -1]))), 0.5)
        self.assertEqual(float(F.auroc(torch.tensor([-0.0, 0.0]), torch.tensor([0, -1]))), 0.5)

    def test_one_class_raises(self):
        for labels in (torch.zeros(5).long(), -torch.ones(5).long()):
            for fn in (F.auroc, F.aupr, F.fpr_at_tpr, F.autc):
                with self.subTest(fn=fn.__name__, label=int(labels[0])):
                    with self.assertRaises(ValueError):
                        fn(torch.arange(5.0), labels)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            F.auroc(torch.zeros(4), torch.zeros(5))
        with self.assertRaises(ValueError):
            F.auroc(torch.zeros(2, 3), torch.zeros(3, 2))

    def test_invalid_arguments(self):
        scores, labels = torch.arange(4.0), torch.tensor([0, 0, -1, -1])
        with self.assertRaises(ValueError):
            F.aupr(scores, labels, positive="out")
        with self.assertRaises(ValueError):
            F.pr_curve(scores, labels, positive="in")
        for tpr in (-0.1, 1.5, 95):
            with self.assertRaises(ValueError):
                F.fpr_at_tpr(scores, labels, tpr)

    def test_does_not_modify_inputs(self):
        scores, labels = _data(0, 100)
        before = scores.clone(), labels.clone()
        _ours(scores, labels)
        F.autc(scores, labels)
        self.assertTrue(torch.equal(scores, before[0]))
        self.assertTrue(torch.equal(labels, before[1]))

    def test_requires_grad(self):
        scores = torch.randn(20, requires_grad=True)
        labels = torch.tensor([0] * 10 + [-1] * 10)
        self.assertFalse(F.auroc(scores, labels).requires_grad)

    def test_permutation_invariant(self):
        scores, labels = _data(1, 3000, tie_levels=20)
        expected = _ours(scores, labels)
        for seed in range(5):
            perm = torch.randperm(3000, generator=torch.Generator().manual_seed(seed))
            self.assertEqual(_ours(scores[perm], labels[perm]), expected)

    def test_invariant_to_increasing_maps(self):
        # integer-valued scores, so the maps below are exact and create no new ties
        scores, labels = _data(2, 3000, tie_levels=500)
        expected = _ours(scores, labels)
        for transform in (
            lambda s: s * 1000 + 1e6,
            lambda s: s - 1e6,
            lambda s: s**3,
            lambda s: s.exp(),
        ):
            self.assertEqual(_ours(transform(scores), labels), expected)
        self.assertAlmostEqual(
            float(F.autc(scores * 1000 + 1e6, labels)), float(F.autc(scores, labels)), places=12
        )

    def test_fpr_at_tpr_conventions(self):
        # 20 OOD samples, so a true positive rate of 0.95 is reached exactly at 19 of them
        ood = torch.arange(20.0) + 10.5
        ind = torch.arange(30.0)
        scores = torch.cat([ind, ood])
        labels = torch.cat([torch.zeros(30), -torch.ones(20)]).long()
        # the 19th highest OOD score is 11.5, above 18 of the ID scores (0..29 > 11.5: 12..29)
        self.assertEqual(float(F.fpr_at_tpr(scores, labels, 0.95)), 18 / 30)
        self.assertEqual(float(F.fpr_at_tpr(scores, labels, 0.0)), 0.0)
        # all OOD samples detected at 10.5: the ID scores 11..29 are above
        self.assertEqual(float(F.fpr_at_tpr(scores, labels, 1.0)), 19 / 30)

    def test_auroc_is_mann_whitney(self):
        scores, labels = _data(3, 300, tie_levels=15)
        ood, ind = scores[labels < 0], scores[labels >= 0]
        greater = sum(Fraction(int((o > ind).sum())) for o in ood)
        ties = sum(Fraction(int((o == ind).sum())) for o in ood)
        expected = (greater + ties / 2) / (len(ood) * len(ind))
        self.assertAlmostEqual(float(F.auroc(scores, labels)), float(expected), places=12)

    def test_more_samples_than_float32_counts(self):
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
        for device in DEVICES:
            with self.subTest(device=device):
                s, y = scores.to(device), labels.to(device)
                counts = F._Counts(s, y < 0)
                self.assertEqual(counts.fps.tolist(), [2001, 2001, n_id, n_id])
                self.assertEqual(counts.tps.tolist(), [0, 1001, 1001, n_ood])
                self.assertAlmostEqual(float(F.auroc(s, y)), float(expected), places=12)
                self.assertEqual(float(F.fpr_at_tpr(s, y, 0.5)), 2001 / n_id)
                # the same with the classes swapped, so that the OOD counts are large
                counts = F._Counts(-s, y >= 0)
                self.assertEqual(counts.tps.tolist(), [0, n_low, n_low, n_id])
                self.assertEqual(counts.fps.tolist(), [1000, 1000, 2001, 2001])


class TestAUTC(unittest.TestCase):
    def test_perfect_separation(self):
        labels = torch.cat([torch.zeros(50), -torch.ones(50)]).long()
        scores = torch.cat([torch.zeros(50), torch.ones(50)])
        self.assertEqual(float(F.autc(scores, labels)), 0.0)

    def test_near_worse_than_far(self):
        g = torch.Generator().manual_seed(0)
        ind = torch.rand(9000, generator=g)
        labels = torch.cat([torch.zeros(9000), -torch.ones(1000)]).long()
        near = F.autc(torch.cat([ind, torch.rand(1000, generator=g) + 2]), labels)
        far = F.autc(torch.cat([ind, torch.rand(1000, generator=g) + 10]), labels)
        self.assertGreater(float(near), float(far))

    @unittest.skipUnless(SKLEARN, "scikit-learn is the reference")
    def test_matches_threshold_curve(self):
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
        self.assertAlmostEqual(float(F.autc(scores, labels)), reference, places=3)

    def test_constant_scores_raise(self):
        with self.assertRaises(ValueError):
            F.autc(torch.ones(10), torch.tensor([0] * 5 + [-1] * 5))

    def test_infinite_scores_raise(self):
        with self.assertRaises(ValueError):
            F.autc(torch.tensor([0.0, 1.0, float("inf")]), torch.tensor([0, -1, -1]))


class TestAccuracy(unittest.TestCase):
    def test_ignores_ood(self):
        predictions = torch.tensor([1, 2, 3, 0, 0])
        labels = torch.tensor([1, 2, 0, -1, -1])
        self.assertAlmostEqual(float(F.accuracy(predictions, labels)), 2 / 3)

    def test_large_class_indices(self):
        labels = torch.tensor([70000, 1000, 5])
        self.assertEqual(float(F.accuracy(labels.clone(), labels)), 1.0)

    def test_no_id_raises(self):
        with self.assertRaises(ValueError):
            F.accuracy(torch.zeros(3).long(), -torch.ones(3).long())

    @unittest.skipUnless(SKLEARN, "scikit-learn is the reference")
    def test_matches_sklearn(self):
        g = torch.Generator().manual_seed(0)
        labels = torch.randint(-1, 10, (1000,), generator=g)
        predictions = torch.randint(0, 10, (1000,), generator=g)
        known = labels >= 0
        expected = skm.accuracy_score(labels[known].numpy(), predictions[known].numpy())
        self.assertAlmostEqual(float(F.accuracy(predictions, labels)), expected, places=12)


class TestMovedFunctions(unittest.TestCase):
    def test_calibration_error(self):
        conf = torch.linspace(0, 1, 1000)
        y = torch.ones(1000)
        y[500:] = 0
        self.assertGreater(calibration_error(conf, y), 0)

    def test_aurra(self):
        self.assertEqual(aurra(torch.tensor([0.9, 0.1]), torch.tensor([1, 0])), 0.75)

    def test_openness(self):
        self.assertAlmostEqual(
            calc_openness(n_train=6, n_test=10, n_target=6), 1 - float(np.sqrt(12 / 16))
        )

    def test_openness_of_closed_set_is_zero(self):
        self.assertEqual(calc_openness(n_train=6, n_test=6, n_target=6), 0)
