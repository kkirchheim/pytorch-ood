import unittest

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import ModelNotSetException, RequiresFittingException
from src.pytorch_ood.detector import GEN, CADRef, CARef, EnergyBased, MaxLogit, MaxSoftmax
from src.pytorch_ood.utils import OODMetrics
from tests.helpers import ClassificationModel, sample_dataset


def ref_energy(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits, dim=1)


def ref_max_logit(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(dim=1).values


def ref_msp(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).max(dim=1).values


def ref_gen(logits: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    m = max(10, logits.shape[-1] // 10)
    p = F.softmax(logits, dim=1)
    top = torch.sort(p, dim=1).values[:, -m:]
    return 1.0 / torch.sum(top**gamma * (1 - top) ** gamma, dim=1)


def ref_gen_untruncated(logits: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """The same generalized entropy, but over all classes. Only used to show that the
    truncation in :func:`ref_gen` is not a no-op for the label space under test."""
    p = F.softmax(logits, dim=1)
    return 1.0 / torch.sum(p**gamma * (1 - p) ** gamma, dim=1)


#: Confidences transcribed from the reference implementation, deliberately *not* imported from
#: the module under test so that the comparison below stays independent of it.
REFERENCE_CONFIDENCES = {
    "energy": ref_energy,
    "maxlogit": ref_max_logit,
    "msp": ref_msp,
    "gen": ref_gen,
}

#: The library score function that is expected to reproduce each reference confidence once
#: CADRef negates it. Pairing these is the point of ``test_matches_reference_implementation``.
PRODUCTION_SCORES = {
    "energy": EnergyBased.score,
    "maxlogit": MaxLogit.score,
    "msp": MaxSoftmax.score,
    "gen": CADRef.gen_score,
}


def constant_score(value: float):
    """A ``logit_score`` whose *confidence* (its negation) is exactly ``value``."""
    return lambda logits: torch.full((logits.shape[0],), -float(value))


def reference_scores(
    z_fit: torch.Tensor,
    z_test: torch.Tensor,
    head: torch.nn.Linear,
    confidence,
) -> torch.Tensor:
    """
    Independent transcription of the reference implementation
    (https://github.com/LingAndZero/CADRef, ``ood_methods/CADRef.py``), written with explicit
    Python loops instead of vectorized tensor ops so that it does not share code paths with
    the implementation under test.
    """
    with torch.no_grad():
        logits_fit = head(z_fit)
        w = head.weight.detach()

        n_classes = logits_fit.shape[1]
        means = []
        for k in range(n_classes):
            members = [z_fit[i] for i in range(len(z_fit)) if logits_fit[i].argmax().item() == k]
            means.append(torch.stack(members).mean(dim=0) if members else z_fit.mean(dim=0))

        mean_confidence = confidence(logits_fit).mean().item()

        logits_test = head(z_test)
        scores = []
        for i in range(len(z_test)):
            c = int(logits_test[i].argmax().item())
            e_p, e_n = 0.0, 0.0
            for j in range(z_test.shape[1]):
                d = (z_test[i, j] - means[c][j]).item()
                s = float(torch.sign(w[c, j]).item())
                if d * s > 0:
                    e_p += abs(d)
                elif d * s < 0:
                    e_n += abs(d)
            norm = z_test[i].abs().sum().item()
            s_logit = confidence(logits_test[i].unsqueeze(0)).item()
            scores.append(e_p / norm / s_logit + e_n / norm / mean_confidence)

        return torch.tensor(scores)


class TestCADRef(unittest.TestCase):
    """
    Tests for the CADRef detector
    """

    def _train_model(self, loader, num_outputs=3, n_hidden=128):
        model = ClassificationModel(num_outputs=num_outputs, n_hidden=n_hidden)
        sgd = SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9, nesterov=True)
        for _ in range(20):
            for x, y in loader:
                sgd.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                sgd.step()
        return model.eval()

    @staticmethod
    def _positive_head() -> torch.nn.Linear:
        """A head whose energy is comfortably positive, so that the scaling stays well-defined"""
        head = torch.nn.Linear(4, 3)
        with torch.no_grad():
            head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -1.0, 2.0, -0.5],
                        [-2.0, 1.0, 0.5, 1.0],
                        [0.5, 0.5, -1.0, 2.0],
                    ]
                )
            )
            head.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))
        return head

    def test_requires_fit_flag(self):
        self.assertTrue(CADRef.requires_fit)

    def test_extends_caref(self):
        """CADRef refines CARef and reuses its class-centroid fitting"""
        self.assertTrue(issubclass(CADRef, CARef))

    def test_nofitting(self):
        model = ClassificationModel().eval()
        detector = CADRef(encoder=model.features, head=model.classifier)

        with self.assertRaises(RequiresFittingException):
            detector(torch.randn(size=(16, 10)))

        with self.assertRaises(RequiresFittingException):
            detector.predict_features(torch.randn(size=(16, 10)))

    def test_no_encoder(self):
        model = ClassificationModel().eval()
        detector = CADRef(encoder=None, head=model.classifier)
        detector.fit_features(torch.randn(size=(64, 10)))

        with self.assertRaises(ModelNotSetException):
            detector(torch.randn(size=(16, 10)))

        with self.assertRaises(ModelNotSetException):
            detector.fit(DataLoader(TensorDataset(torch.randn(16, 10), torch.zeros(16).long())))

    def test_head_must_expose_weights(self):
        with self.assertRaises(ValueError):
            CADRef(encoder=None, head=None)

        with self.assertRaises(ValueError):
            CADRef(encoder=None, head=torch.nn.ReLU())

        # has a .weight, but not a (num_classes, num_features) matrix
        with self.assertRaises(ValueError):
            CADRef(encoder=None, head=torch.nn.Conv2d(3, 4, kernel_size=3))

    def test_logit_score_is_negated_into_a_confidence(self):
        """``logit_score`` is a library outlier score; its negation is the divisor.

        Pinning this catches a dropped or doubled negation, which would silently invert the
        error scaling while leaving every shape and finiteness check happy.
        """
        torch.manual_seed(17)
        head = self._positive_head()
        z_fit = torch.randn(size=(128, 4))
        z_test = torch.randn(size=(16, 4))

        detector = CADRef(encoder=None, head=head, logit_score=EnergyBased.score)
        detector.fit_features(z_fit)

        with torch.no_grad():
            logits_fit, logits_test = head(z_fit), head(z_test)
        # EnergyBased.score is -logsumexp, so the confidence must come out as +logsumexp
        self.assertAlmostEqual(
            detector.mean_logit_score.item(),
            torch.logsumexp(logits_fit, dim=1).mean().item(),
            places=5,
        )

        caref_error = CARef(encoder=None, head=head)
        caref_error.train_means = detector.train_means
        expected_total = caref_error.predict_features(z_test)

        scores = detector.predict_features(z_test)
        # every component is positive, so the scaled score must lie between the two extremes
        # obtained by dividing the whole CARef error by one or the other confidence
        confidence = torch.logsumexp(logits_test, dim=1)
        lo = torch.minimum(expected_total / confidence, expected_total / detector.mean_logit_score)
        hi = torch.maximum(expected_total / confidence, expected_total / detector.mean_logit_score)
        self.assertTrue(((scores >= lo - 1e-6) & (scores <= hi + 1e-6)).all())

    def test_gen_score_is_not_interchangeable_with_gen_detector(self):
        """``GEN.score`` must not be silently usable in place of :meth:`CADRef.gen_score`"""
        torch.manual_seed(3)
        logits = torch.randn(size=(32, 60)) * 3

        self.assertFalse(torch.allclose(CADRef.gen_score(logits), GEN.score(logits)))
        # gen_score is a valid library score: it ranks identically to the generalized entropy
        self.assertTrue(
            torch.equal(
                CADRef.gen_score(logits).argsort(),
                ref_gen(logits).reciprocal().argsort(),
            )
        )
        # and negating it recovers the reference divisor exactly
        self.assertTrue(torch.allclose(-CADRef.gen_score(logits), ref_gen(logits), atol=1e-6))

    def test_rejects_grid_input(self):
        detector = CADRef(encoder=None, head=self._positive_head())

        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(size=(16, 4, 4, 4)))

    def test_score_shape_and_finiteness(self):
        torch.manual_seed(3)
        detector = CADRef(encoder=None, head=self._positive_head())
        detector.fit_features(torch.randn(size=(256, 4)))

        scores = detector.predict_features(torch.randn(size=(32, 4)))

        self.assertEqual(scores.shape, (32,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_matches_reference_implementation(self):
        """Compare against a loop-based transcription of the official implementation"""
        for name in ["energy", "maxlogit", "msp", "gen"]:
            with self.subTest(logit_score=name):
                torch.manual_seed(31)
                head = self._positive_head()
                z_fit = torch.randn(size=(128, 4)) + 1.0
                z_test = torch.randn(size=(24, 4)) * 2.0

                detector = CADRef(encoder=None, head=head, logit_score=PRODUCTION_SCORES[name])
                detector.fit_features(z_fit)

                expected = reference_scores(z_fit, z_test, head, REFERENCE_CONFIDENCES[name])
                self.assertTrue(
                    torch.allclose(detector.predict_features(z_test), expected, atol=1e-5)
                )

    def test_matches_reference_implementation_many_classes(self):
        """
        Same comparison with a label space large enough that the top-M truncation of the
        generalized entropy actually engages. With C = 60 the reference keeps M = 10 of 60
        classes, which distinguishes the exact truncation rule from the obvious variants
        (no truncation, a different floor, a different divisor, bottom-M instead of top-M).
        """
        for name in ["gen", "energy", "msp"]:
            with self.subTest(logit_score=name):
                torch.manual_seed(37)
                n_classes, n_features = 60, 8
                head = torch.nn.Linear(n_features, n_classes)
                z_fit = torch.randn(size=(256, n_features)) + 0.5
                z_test = torch.randn(size=(24, n_features)) * 1.5

                detector = CADRef(encoder=None, head=head, logit_score=PRODUCTION_SCORES[name])
                detector.fit_features(z_fit)

                expected = reference_scores(z_fit, z_test, head, REFERENCE_CONFIDENCES[name])
                self.assertTrue(
                    torch.allclose(detector.predict_features(z_test), expected, atol=1e-5)
                )

    def test_gen_truncation_engages_for_large_label_spaces(self):
        """
        Guards the guard: with C = 60 the truncation must actually discard classes, otherwise
        the comparison above would be vacuous.
        """
        torch.manual_seed(41)
        logits = torch.randn(size=(8, 60)) * 3

        truncated = -CADRef.gen_score(logits)
        untruncated = ref_gen_untruncated(logits)

        self.assertFalse(torch.allclose(truncated, untruncated))

    def test_decomposition_sums_to_caref_error(self):
        """
        The positive and negative errors partition the CARef error exactly. Using a constant
        logit confidence of one for both terms must therefore reproduce CARef.
        """
        torch.manual_seed(5)
        head = self._positive_head()
        z_fit = torch.randn(size=(128, 4))
        z_test = torch.randn(size=(32, 4))

        caref = CARef(encoder=None, head=head)
        caref.fit_features(z_fit)

        cadref = CADRef(encoder=None, head=head, logit_score=constant_score(1.0))
        cadref.fit_features(z_fit)

        self.assertTrue(
            torch.allclose(
                cadref.predict_features(z_test), caref.predict_features(z_test), atol=1e-5
            )
        )

    def test_exact_score(self):
        """
        Hand-computed score for a single sample, with a head chosen such that the predicted
        class, the sign pattern, and the confidence are all easy to write down.
        """
        head = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, -1.0, 0.0], [2.0, 2.0, 2.0]]))
            head.bias.copy_(torch.tensor([0.0, 100.0]))  # class 1 always wins

        # both fitting samples are assigned to class 1, centroid = (2, 3, 4)
        z_fit = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

        detector = CADRef(encoder=None, head=head, logit_score=constant_score(2.0))
        detector.fit_features(z_fit)

        self.assertAlmostEqual(detector.mean_logit_score.item(), 2.0, places=6)

        # relative feature of (5, 1, 4) w.r.t. (2, 3, 4) is (3, -2, 0)
        # w[1] = (2, 2, 2), so signs are all +1: positive part = 3, negative part = 2
        # ||z||_1 = 10 -> E_p = 0.3, E_n = 0.2
        # score = 0.3 / 2 + 0.2 / 2 = 0.25
        score = detector.predict_features(torch.tensor([[5.0, 1.0, 4.0]]))
        self.assertAlmostEqual(score.item(), 0.25, places=6)

    def test_sign_pattern_of_head_matters(self):
        """
        Flipping the sign of the head weights swaps the roles of the positive and negative
        error. Since the two are scaled by different constants here, the score must change
        in an exactly predictable way. This pins down which component is which.
        """
        weight = torch.tensor([[1.0, -1.0, 0.0], [2.0, 2.0, 2.0]])
        bias = torch.tensor([0.0, 100.0])  # class 1 always wins, also after the flip

        def make_head(w: torch.Tensor) -> torch.nn.Linear:
            head = torch.nn.Linear(3, 2, bias=True)
            with torch.no_grad():
                head.weight.copy_(w)
                head.bias.copy_(bias)
            return head

        # centroid of class 1 is (2, 3, 4); relative feature of (5, 1, 4) is (3, -2, 0)
        z_fit = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        z_test = torch.tensor([[5.0, 1.0, 4.0]])

        def score_with(w: torch.Tensor) -> float:
            detector = CADRef(
                encoder=None,
                head=make_head(w),
                logit_score=constant_score(1.0),
            )
            detector.fit_features(z_fit)
            # scale the two components differently to make the swap observable
            detector.mean_logit_score = torch.tensor(2.0)
            return detector.predict_features(z_test).item()

        # w[1] = (2, 2, 2): all signs positive, E_p = 3/10, E_n = 2/10 -> 0.3 / 1 + 0.2 / 2
        self.assertAlmostEqual(score_with(weight), 0.4, places=6)
        # flipped: E_p and E_n swap -> 0.2 / 1 + 0.3 / 2
        self.assertAlmostEqual(score_with(-weight), 0.35, places=6)

    def test_zero_weight_dimensions_are_dropped(self):
        """Feature dimensions with zero weight contribute to neither error component"""
        head = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 1.0, 0.0]]))

        z_fit = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])  # centroid = (1, 1, 1)
        detector = CADRef(encoder=None, head=head, logit_score=constant_score(1.0))
        detector.fit_features(z_fit)

        # relative feature (1, -1, 5); the third dimension has zero weight and is ignored,
        # so E_p = 1, E_n = 1 and ||z||_1 = 2 + 0 + 6 = 8 -> 2 / 8
        score = detector.predict_features(torch.tensor([[2.0, 0.0, 6.0]]))
        self.assertAlmostEqual(score.item(), 0.25, places=6)

    def test_score_sign(self):
        """A sample sitting exactly on its centroid has zero error, everything else is larger"""
        head = self._positive_head()
        torch.manual_seed(2)
        z_fit = torch.randn(size=(64, 4))

        detector = CADRef(encoder=None, head=head)
        detector.fit_features(z_fit)

        logits = head(z_fit)
        # pick a class that was actually predicted, so its centroid is a real mean
        k = int(logits.argmax(dim=1).mode().values.item())
        centroid = detector.train_means[k].unsqueeze(0)

        self.assertAlmostEqual(detector.predict_features(centroid).item(), 0.0, places=6)
        self.assertGreater(detector.predict_features(centroid + 1.0).item(), 0.0)

    def test_no_batch_coupling(self):
        """
        The normalization constant comes from the training data, so scoring a sample alone must
        match scoring it inside a batch that contains an extreme outlier.
        """
        torch.manual_seed(7)
        detector = CADRef(encoder=None, head=self._positive_head())
        detector.fit_features(torch.randn(size=(256, 4)))

        batch = torch.randn(size=(16, 4))
        batch[3] *= 50

        batched = detector.predict_features(batch)
        individually = torch.cat([detector.predict_features(row.unsqueeze(0)) for row in batch])

        self.assertTrue(torch.allclose(batched, individually, atol=1e-5))

    def test_mean_logit_score_comes_from_training_data(self):
        """The constant decay of the negative error is estimated on the fitting data"""
        torch.manual_seed(9)
        head = self._positive_head()
        z_fit = torch.randn(size=(128, 4))

        detector = CADRef(encoder=None, head=head, logit_score=EnergyBased.score)
        detector.fit_features(z_fit)

        with torch.no_grad():
            expected = torch.logsumexp(head(z_fit), dim=1).mean()

        self.assertAlmostEqual(detector.mean_logit_score.item(), expected.item(), places=5)

    def test_fit_ignores_ood_samples(self):
        torch.manual_seed(17)
        head = self._positive_head()
        z_id = torch.randn(size=(256, 4))
        z_ood = torch.randn(size=(256, 4)) * 20 + 50

        clean = CADRef(encoder=None, head=head)
        clean.fit_features(z_id, torch.arange(256) % 3)

        mixed = CADRef(encoder=None, head=head)
        mixed.fit_features(
            torch.cat([z_id, z_ood]),
            torch.cat([torch.arange(256) % 3, -torch.ones(256).long()]),
        )

        self.assertTrue(torch.allclose(clean.train_means, mixed.train_means, atol=1e-5))
        self.assertAlmostEqual(
            clean.mean_logit_score.item(), mixed.mean_logit_score.item(), places=4
        )

    def test_fit_without_id_samples(self):
        detector = CADRef(encoder=None, head=self._positive_head())

        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(size=(32, 4)), -torch.ones(32).long())

    def test_fit_chunking_is_exact(self):
        torch.manual_seed(19)
        head = self._positive_head()
        z = torch.randn(size=(200, 4))

        whole = CADRef(encoder=None, head=head)
        whole.fit_features(z, batch_size=1000)

        chunked = CADRef(encoder=None, head=head)
        chunked.fit_features(z, batch_size=7)

        self.assertTrue(torch.allclose(whole.train_means, chunked.train_means, atol=1e-6))
        self.assertAlmostEqual(
            whole.mean_logit_score.item(), chunked.mean_logit_score.item(), places=5
        )

    def test_fit_matches_fit_features(self):
        torch.manual_seed(11)
        model = ClassificationModel().eval()
        x = torch.randn(size=(256, 10))
        loader = DataLoader(TensorDataset(x, torch.arange(256) % 3), batch_size=32)

        from_loader = CADRef(encoder=model.features, head=model.classifier)
        from_loader.fit(loader)

        with torch.no_grad():
            z = model.features(x)
        from_features = CADRef(encoder=model.features, head=model.classifier)
        from_features.fit_features(z)

        self.assertTrue(
            torch.allclose(from_loader.train_means, from_features.train_means, atol=1e-5)
        )
        self.assertAlmostEqual(
            from_loader.mean_logit_score.item(), from_features.mean_logit_score.item(), places=5
        )

    def test_predict_matches_predict_features(self):
        torch.manual_seed(13)
        model = ClassificationModel().eval()
        x = torch.randn(size=(128, 10))

        detector = CADRef(encoder=model.features, head=model.classifier)
        with torch.no_grad():
            detector.fit_features(model.features(x))

        x_test = torch.randn(size=(16, 10))
        with torch.no_grad():
            expected = detector.predict_features(model.features(x_test))

        self.assertTrue(torch.allclose(detector.predict(x_test), expected, atol=1e-6))

    def test_logit_scores_differ(self):
        """The choice of the logit score must actually influence the outcome"""
        torch.manual_seed(23)
        head = self._positive_head()
        z_fit = torch.randn(size=(128, 4))
        z_test = torch.randn(size=(16, 4))

        scores = {}
        for name in ["energy", "maxlogit", "msp", "gen"]:
            detector = CADRef(encoder=None, head=head, logit_score=PRODUCTION_SCORES[name])
            detector.fit_features(z_fit)
            scores[name] = detector.predict_features(z_test)
            self.assertTrue(torch.isfinite(scores[name]).all())

        self.assertFalse(torch.allclose(scores["energy"], scores["msp"]))
        self.assertFalse(torch.allclose(scores["energy"], scores["gen"]))

    def test_hyperparameters(self):
        detector = CADRef(encoder=None, head=self._positive_head())

        self.assertEqual(sorted(CADRef.hyperparameter_space), ["logit_score"])
        self.assertEqual(
            CADRef.hyperparameter_space["logit_score"],
            [EnergyBased.score, CADRef.gen_score, MaxLogit.score, MaxSoftmax.score],
        )
        self.assertEqual(detector.get_hyperparameters(), {"logit_score": EnergyBased.score})

        detector.set_hyperparameters(logit_score=CADRef.gen_score)
        self.assertEqual(detector.logit_score, CADRef.gen_score)

        with self.assertRaises(ValueError):
            detector.set_hyperparameters(unknown=1)

    def test_grid_search(self):
        """CADRef can be tuned with the grid search utility, which refits per candidate"""
        from src.pytorch_ood.utils import GridSearch

        torch.manual_seed(42)
        fit_set = sample_dataset(n_samples=200, n_dim=10, centers=3, seed=42, loc=3)
        fit_loader = DataLoader(fit_set, batch_size=200)
        model = self._train_model(fit_loader)

        val_id = sample_dataset(n_samples=50, n_dim=10, centers=3, seed=1, loc=3)
        val_ood = sample_dataset(n_samples=50, n_dim=10, centers=3, std=5, seed=2, loc=-3)
        val_x = torch.cat([val_id.tensors[0], val_ood.tensors[0]])
        val_y = torch.cat([val_id.tensors[1], -torch.ones(len(val_ood)).long()])
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=100)

        detector = CADRef(encoder=model.features, head=model.classifier)
        search = GridSearch(
            detector,
            fit_loader=fit_loader,
            val_loader=val_loader,
            hyperparameter_space={"logit_score": [EnergyBased.score, MaxSoftmax.score]},
        )
        best = search.run()

        self.assertIn(best["logit_score"], [EnergyBased.score, MaxSoftmax.score])
        self.assertEqual(detector.logit_score, best["logit_score"])

    def test_repr(self):
        detector = CADRef(encoder=None, head=self._positive_head(), logit_score=CADRef.gen_score)
        self.assertEqual(repr(detector), "CADRef(logit_score=_gen_score)")

        self.assertEqual(
            repr(CADRef(encoder=None, head=self._positive_head())),
            "CADRef(logit_score=EnergyBased.score)",
        )

    def test_detects_outliers(self):
        torch.manual_seed(42)
        dataset = sample_dataset(n_samples=1000, n_dim=10, centers=3, seed=42, loc=3)
        loader = DataLoader(dataset, batch_size=1000)
        model = self._train_model(loader)

        detector = CADRef(encoder=model.features, head=model.classifier)
        detector.fit(loader)

        x, y = next(iter(loader))
        scores_in = detector.predict(x)

        ood = sample_dataset(n_samples=100, n_dim=10, centers=3, std=5, seed=42, loc=-3)
        x_ood, _ = next(iter(DataLoader(ood, batch_size=1000)))
        scores_out = detector.predict(x_ood)

        metrics = OODMetrics()
        metrics.update(scores_in, y)
        metrics.update(scores_out, -torch.ones_like(scores_out).long())

        auroc = metrics.compute()["AUROC"]
        print(f"AUROC: {auroc}")
        self.assertGreater(auroc, 0.8)

    def test_scores_are_graph_free(self):
        """Scores must never carry an autograd graph back to the caller"""
        detector = CADRef(encoder=None, head=self._positive_head())
        detector.fit_features(torch.randn(size=(64, 4), requires_grad=True))

        scores = detector.predict_features(torch.randn(size=(8, 4), requires_grad=True))

        self.assertFalse(scores.requires_grad)
        self.assertIsNone(scores.grad_fn)
        self.assertFalse(detector.train_means.requires_grad)
        self.assertFalse(detector.mean_logit_score.requires_grad)

    def test_fit_features_rejects_empty_input(self):
        detector = CADRef(encoder=None, head=self._positive_head())

        with self.assertRaises(ValueError):
            detector.fit_features(torch.zeros(size=(0, 4)))

    def test_features_are_cast_to_float32(self):
        """float64 input is downcast so a standard float32 head can consume it"""
        detector = CADRef(encoder=None, head=self._positive_head())
        detector.fit_features(torch.randn(size=(32, 4), dtype=torch.float64))

        self.assertEqual(detector.train_means.dtype, torch.float32)

        scores = detector.predict_features(torch.randn(size=(8, 4), dtype=torch.float64))
        self.assertEqual(scores.dtype, torch.float32)

    def test_small_norm_features_are_not_clamped_away(self):
        """
        The normalization guard must only protect against a zero norm; a small but nonzero
        norm has to be used as-is.
        """
        head = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, -1.0, 0.0], [2.0, 2.0, 2.0]]))
            head.bias.copy_(torch.tensor([0.0, 100.0]))  # class 1 always wins

        # centroid of class 1 is (0.02, 0.03, 0.04)
        z_fit = torch.tensor([[0.01, 0.02, 0.03], [0.03, 0.04, 0.05]])

        detector = CADRef(encoder=None, head=head, logit_score=constant_score(1.0))
        detector.fit_features(z_fit)

        # relative feature of (0.05, 0.01, 0.04) is (0.03, -0.02, 0.0), all weights positive
        # -> E_p = 0.03 / 0.1, E_n = 0.02 / 0.1, both divided by 1 -> 0.5
        score = detector.predict_features(torch.tensor([[0.05, 0.01, 0.04]]))
        self.assertAlmostEqual(score.item(), 0.5, places=5)

    def test_zero_feature_vector_is_finite(self):
        detector = CADRef(encoder=None, head=self._positive_head())
        detector.fit_features(torch.randn(size=(64, 4)))

        score = detector.predict_features(torch.zeros(size=(1, 4)))
        self.assertTrue(torch.isfinite(score).all())

    def test_state_follows_to_device(self):
        detector = CADRef(encoder=None, head=self._positive_head())
        detector.fit_features(torch.randn(size=(64, 4)))

        detector.to("cpu")

        self.assertEqual(detector.train_means.device.type, "cpu")
        self.assertEqual(detector.mean_logit_score.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_fit_features_moves_cpu_features_to_detector_device(self):
        """
        fit_features() is not wrapped by the base class, so it must move CPU features onto the
        detector's device itself -- for both pieces of fitted state.
        """
        model = ClassificationModel().eval().cuda()
        detector = CADRef(encoder=model.features, head=model.classifier).to("cuda")

        detector.fit_features(torch.randn(size=(256, 10)))  # deliberately on the CPU

        self.assertEqual(detector.train_means.device.type, "cuda")
        self.assertEqual(detector.mean_logit_score.device.type, "cuda")

        scores = detector.predict_features(torch.randn(size=(8, 10)))
        self.assertEqual(scores.device.type, "cuda")
        self.assertTrue(torch.isfinite(scores).all())

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_cuda(self):
        model = ClassificationModel().eval().cuda()
        detector = CADRef(encoder=model.features, head=model.classifier).to("cuda")
        detector.fit_features(torch.randn(size=(256, 10)).cuda())

        self.assertEqual(detector.train_means.device.type, "cuda")
        self.assertEqual(detector.mean_logit_score.device.type, "cuda")

        scores = detector.predict(torch.randn(size=(16, 10)).cuda())

        self.assertEqual(scores.device.type, "cuda")
        self.assertTrue(torch.isfinite(scores).all())
