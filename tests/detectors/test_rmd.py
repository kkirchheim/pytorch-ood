import unittest

import torch

from src.pytorch_ood.detector import RMD


class RMDBackgroundTest(unittest.TestCase):
    def test_background_covariance_is_normalized(self):
        """The background Gaussian uses the covariance of all known features (divided by N)."""
        torch.manual_seed(0)
        y = torch.arange(40) % 2
        z = torch.randn(40, 10) + 3 * y.unsqueeze(1)
        detector = RMD(None).fit_features(z, y)
        centered = z.double() - z.double().mean(dim=0)
        expected = centered.T @ centered / len(z) + 1e-10 * torch.eye(10, dtype=torch.float64)
        torch.testing.assert_close(detector.background_cov, expected.float())
        torch.testing.assert_close(
            detector.background_precision, torch.linalg.inv(expected).float()
        )
