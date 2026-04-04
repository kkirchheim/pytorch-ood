import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import RankFeat
from src.pytorch_ood.detector.rankfeat import _remove_rank1
from pytorch_ood.utils import OODMetrics


class SimpleCNN(nn.Module):
    """Tiny CNN that exposes backbone/head split for testing."""

    def __init__(self, num_classes=3):
        super().__init__()
        # backbone: outputs (B, 8, 4, 4)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        # head: pool + linear
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def backbone(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def head(self, x):
        x = self.gap(x).flatten(1)
        return self.fc(x)

    def forward(self, x):
        return self.head(self.backbone(x))


class TestRemoveRank1(unittest.TestCase):
    """Unit tests for the _remove_rank1 helper."""

    def test_output_shape(self):
        x = torch.randn(4, 8, 6, 6)
        out = _remove_rank1(x)
        self.assertEqual(out.shape, x.shape)

    def test_rank1_removed(self):
        """
        Construct a rank-1 matrix plus noise. After removal the leading
        singular value should drop significantly.
        """
        B, C, HW = 2, 8, 16
        # Dominant rank-1 component
        u = torch.randn(B, C, 1)
        v = torch.randn(B, 1, HW)
        rank1 = 10.0 * u.bmm(v)
        noise = torch.randn(B, C, HW) * 0.1
        x = (rank1 + noise).view(B, C, 4, 4)

        s_before = torch.linalg.svdvals(x.view(B, C, HW))
        out = _remove_rank1(x)
        s_after = torch.linalg.svdvals(out.view(B, C, HW))

        # The leading singular value should drop by at least 10x
        for i in range(B):
            self.assertLess(s_after[i, 0].item(), s_before[i, 0].item() * 0.1)

    def test_preserves_gradient(self):
        """Operation should be differentiable."""
        x = torch.randn(2, 4, 3, 3, requires_grad=True)
        out = _remove_rank1(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestRankFeat(unittest.TestCase):
    """Tests for the RankFeat OOD detector."""

    def test_output_shape(self):
        model = SimpleCNN(num_classes=5).eval()
        detector = RankFeat(backbone=model.backbone, head=model.head)
        x = torch.randn(8, 3, 8, 8)
        with torch.no_grad():
            scores = detector(x)
        self.assertEqual(scores.shape, (8,))

    def test_scores_finite(self):
        model = SimpleCNN().eval()
        detector = RankFeat(backbone=model.backbone, head=model.head)
        x = torch.randn(16, 3, 8, 8)
        with torch.no_grad():
            scores = detector(x)
        self.assertTrue(torch.isfinite(scores).all())

    def test_custom_detector(self):
        """Can use a custom scoring function instead of energy."""
        model = SimpleCNN(num_classes=3).eval()

        def max_logit_score(logits):
            return -logits.max(dim=1).values

        detector = RankFeat(
            backbone=model.backbone,
            head=model.head,
            detector=max_logit_score,
        )
        x = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            scores = detector(x)
        self.assertEqual(scores.shape, (4,))

    def test_fit_is_noop(self):
        model = SimpleCNN().eval()
        detector = RankFeat(backbone=model.backbone, head=model.head)
        result = detector.fit(None)
        self.assertIs(result, detector)

    def test_scores_differ_from_plain_energy(self):
        """
        RankFeat should produce different scores than plain energy on the same
        inputs, confirming that the rank-1 removal actually changes the logits.
        """
        model = SimpleCNN(num_classes=3).eval()
        rankfeat = RankFeat(backbone=model.backbone, head=model.head)

        x = torch.randn(16, 3, 8, 8)
        with torch.no_grad():
            scores_rf = rankfeat(x)
            # Plain energy without rank-1 removal
            logits_plain = model(x)
            from src.pytorch_ood.detector import EnergyBased

            scores_plain = EnergyBased.score(logits_plain)

        self.assertFalse(
            torch.allclose(scores_rf, scores_plain, atol=1e-5),
            "RankFeat scores should differ from plain energy scores",
        )


if __name__ == "__main__":
    unittest.main()
