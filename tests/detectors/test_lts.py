import unittest

import numpy as np
import torch
import torch.nn.functional as F

from src.pytorch_ood.detector import LTS
from src.pytorch_ood.detector.energy import EnergyBased
from tests.helpers import ClassificationModel


def _official_lts(x, percentile=65):
    # verbatim from https://github.com/andrijazz/lts (lts.py), which returns the factor that
    # multiplies the logits: ``get_score(logits * s, ...)`` in ood_eval.py
    assert x.dim() == 2
    assert 0 <= percentile <= 100

    x = F.relu(x)
    s1 = x.sum(dim=1)
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    s2 = v.sum(dim=1)
    scale = s1 / s2
    return scale[:, None] ** 2


class TestLTS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ClassificationModel(num_inputs=10, n_hidden=8, num_outputs=3)
        self.model.eval()

    def test_predict_returns_correct_shape(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier, p=0.05)
        x = torch.randn(16, 10)
        scores = detector(x)
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_predict_features_returns_correct_shape(self):
        detector = LTS(encoder=None, head=self.model.classifier, p=0.05)
        features = torch.randn(16, 8)
        scores = detector.predict_features(features)
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_predict_without_encoder_raises(self):
        detector = LTS(encoder=None, head=self.model.classifier)
        x = torch.randn(4, 10)
        with self.assertRaises(Exception):
            detector.predict(x)

    def test_scaling_factor_shape(self):
        z = torch.randn(8, 64)
        s = LTS.scaling_factor(z, p=0.05)
        self.assertEqual(s.shape, (8,))
        self.assertTrue(torch.isfinite(s).all())

    def test_scaling_factor_uniform_activations(self):
        """When all activations are equal, S1/S2 = n/k, so the factor is (n/k)^2."""
        z = torch.ones(4, 100)
        s = LTS.scaling_factor(z, p=0.05)
        torch.testing.assert_close(s, torch.full((4,), (100.0 / 5) ** 2))

    def test_scaling_factor_concentrated_activations(self):
        """When all mass is in the top p%, S1 = S2, so the factor is 1."""
        z = torch.zeros(4, 100)
        z[:, :5] = 100.0
        s = LTS.scaling_factor(z, p=0.05)
        torch.testing.assert_close(s, torch.ones(4))

    def test_p_equals_one_means_no_scaling(self):
        """With p=1.0, the top 100% are all activations, so the factor is 1."""
        z = torch.randn(4, 64).abs()
        s = LTS.scaling_factor(z, p=1.0)
        torch.testing.assert_close(s, torch.ones(4))

    def test_scaling_factor_ignores_negative_features(self):
        z = torch.tensor([[1.0, 1.0, 1.0, 1.0, -3.0, -5.0]])
        torch.testing.assert_close(
            LTS.scaling_factor(z, p=0.5), LTS.scaling_factor(z.relu(), p=0.5)
        )
        # S1 = 4 and the top 3 activations sum to 3
        torch.testing.assert_close(LTS.scaling_factor(z, p=0.5), torch.tensor([(4 / 3) ** 2]))

    def test_scaling_factor_matches_official_code(self):
        torch.manual_seed(0)
        for p in (0.05, 0.1, 0.35):
            for n in (8, 10, 64, 100, 512, 2048):
                with self.subTest(p=p, n=n):
                    z = torch.randn(16, n)
                    expected = _official_lts(z, percentile=round(100 * (1 - p))).squeeze(1)
                    if not torch.isfinite(expected).all():
                        continue  # the official code selects k = 0 here, guarded in LTS
                    torch.testing.assert_close(LTS.scaling_factor(z, p), expected)

    def test_scaling_factor_at_least_one_activation(self):
        # the official code would select k = 0 here (10 - round(9.5) = 0)
        z = torch.rand(4, 10)
        s = LTS.scaling_factor(z, p=0.05)
        torch.testing.assert_close(s, (z.sum(dim=1) / z.max(dim=1).values) ** 2)

    def test_scaling_factor_rejects_feature_maps(self):
        with self.assertRaises(ValueError):
            LTS.scaling_factor(torch.rand(4, 16, 8, 8), p=0.05)

    def test_score_multiplies_logits(self):
        """Energy of the logits multiplied by S, as in the paper and the official code."""
        head = torch.nn.Linear(8, 3)
        z = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        # k = 8 - round(0.75 * 8) = 2, so S = (4 / 2)^2 = 4
        detector = LTS(encoder=None, head=head, p=0.25)
        with torch.no_grad():
            expected = -torch.logsumexp(4.0 * head(z), dim=1)
        torch.testing.assert_close(detector.predict_features(z), expected)

    def test_matches_official_pipeline(self):
        # wide enough that the official code selects k > 0 for p = 0.05
        model = ClassificationModel(num_inputs=10, n_hidden=512, num_outputs=3).eval()
        detector = LTS(encoder=model.features, head=model.classifier, p=0.05)
        x = torch.randn(16, 10)
        with torch.no_grad():
            features = model.features(x)
            logits = model.classifier(features)
            s = _official_lts(features, percentile=95)
            # the official energy score is larger for ID samples
            expected = -torch.logsumexp(logits * s, dim=1)
        torch.testing.assert_close(detector(x), expected)

    def test_custom_detector_callable(self):
        """LTS should accept any scoring function that maps logits -> scores."""
        seen = []
        detector = LTS(
            encoder=None,
            head=torch.nn.Identity(),
            p=0.5,
            detector=lambda logits: seen.append(logits) or -logits.max(dim=1).values,
        )
        z = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        scores = detector.predict_features(z)
        # k = 2, so S = (8 / 4)^2 = 4, and the detector receives the scaled logits
        torch.testing.assert_close(seen[0], 4 * z)
        self.assertEqual(scores.shape, (1,))

    def test_does_not_require_fit(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier)
        self.assertFalse(detector.requires_fit)

    def test_scores_differ_from_plain_energy(self):
        """LTS should produce different scores than plain energy (unless S=1 everywhere)."""
        x = torch.randn(16, 10)
        with torch.no_grad():
            features = self.model.features(x)
            logits = self.model.classifier(features)

        energy_scores = EnergyBased.score(logits)
        lts_detector = LTS(encoder=self.model.features, head=self.model.classifier, p=0.05)
        lts_scores = lts_detector(x)

        self.assertFalse(torch.allclose(energy_scores, lts_scores))

    def test_batch_size_one(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier)
        x = torch.randn(1, 10)
        scores = detector(x)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_default_p(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier)
        self.assertEqual(detector.p, 0.05)


if __name__ == "__main__":
    unittest.main()
