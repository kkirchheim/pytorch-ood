import unittest

import torch

from pytorch_ood.detector import SCALE
from pytorch_ood.detector.ash import ash_s
from pytorch_ood.detector.scale import scale


class TestSCALE(unittest.TestCase):
    """
    Tests for the SCALE detector.
    """

    def test_does_not_prune(self):
        """
        SCALE keeps all activations (only scales), whereas ASH-S prunes the
        smallest ``percentile`` fraction to zero. So SCALE must have no extra
        zeros beyond those already present in the input.
        """
        x = torch.rand(size=(4, 8, 5, 5)) + 0.1  # strictly positive
        scale_out = scale(x.clone(), percentile=0.65)
        ash_out = ash_s(x.clone(), percentile=0.65)

        # SCALE leaves every position non-zero, ASH-S zeros most of them
        self.assertEqual((scale_out == 0).sum().item(), 0)
        self.assertGreater((ash_out == 0).sum().item(), 0)

    def test_scale_factor_positive(self):
        """Scaling factor exp(s1/s2) >= 1, so output magnitude grows."""
        x = torch.rand(size=(4, 8, 5, 5)) + 0.1
        out = scale(x.clone(), percentile=0.65)
        # since s1 >= s2 > 0 for non-negative activations, scale factor >= 1
        self.assertTrue(torch.all(out.abs() >= x.abs() - 1e-6))

    def test_separates_gaussians(self):
        """
        Synthetic sanity check: ID and OOD features with different norms should
        be separable by the resulting energy score.
        """
        from pytorch_ood.metrics import OODMetrics

        torch.manual_seed(0)
        head = torch.nn.Linear(16, 10)
        detector = SCALE(backbone=None, head=lambda f: head(f.flatten(1)), percentile=0.65)

        # ID: large-norm features, OOD: small-norm features
        z_in = torch.randn(128, 16, 1, 1).abs() * 3.0
        z_out = torch.randn(128, 16, 1, 1).abs() * 0.5

        s_in = detector.predict_feature_maps(z_in)
        s_out = detector.predict_feature_maps(z_out)

        metrics = OODMetrics()
        metrics.update(s_in, torch.zeros(128))  # ID labels >= 0
        metrics.update(s_out, -torch.ones(128).long())  # OOD labels < 0
        auroc = metrics.compute()["AUROC"]
        self.assertGreater(auroc, 0.6)


if __name__ == "__main__":
    unittest.main()
