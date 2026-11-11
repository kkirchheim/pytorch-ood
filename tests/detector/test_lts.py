import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_ood.detector import LTS
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


if __name__ == "__main__":
    unittest.main()
