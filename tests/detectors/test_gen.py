import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import GEN
from tests.helpers import ClassificationModel, SegmentationModel
from pytorch_ood.utils import OODMetrics


def _make_id_ood_data(n_dim=10, n_classes=3, n_per_class=100, ood_offset=20.0, seed=42):
    """Pure-torch helper: well-separated Gaussians for ID, far-offset cluster for OOD."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, n_dim) * 5.0

    x_id = torch.cat(
        [torch.randn(n_per_class, n_dim, generator=g) * 0.3 + centers[c] for c in range(n_classes)]
    )
    y_id = torch.repeat_interleave(torch.arange(n_classes), n_per_class)

    x_ood = torch.randn(n_per_class * n_classes, n_dim, generator=g) * 0.3 + ood_offset
    y_ood = torch.full((n_per_class * n_classes,), -1, dtype=torch.long)

    return x_id, y_id, x_ood, y_ood


def _train_model(model, x_train, y_train, epochs=200, lr=1e-2, seed=42):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x_train), y_train).backward()
        optimizer.step()
    model.eval()
    return model


class TestGEN(unittest.TestCase):
    """Tests for the GEN (Generalized Entropy) OOD detector."""

    def test_output_shape_classification(self):
        """Scalar score per sample for 1-D classification inputs."""
        model = ClassificationModel(num_inputs=10, num_outputs=5)
        detector = GEN(model)
        x = torch.randn(16, 10)
        with torch.no_grad():
            scores = detector(x)
        self.assertEqual(scores.shape, (16,))

    def test_output_shape_segmentation(self):
        """Pixel-wise score for segmentation model (B, H, W) output."""
        model = SegmentationModel(in_channels=3, out_channels=4)
        detector = GEN(model)
        x = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            scores = detector(x)
        self.assertEqual(scores.shape, (4, 8, 8))

    def test_predict_logits(self):
        """predict_logits works on raw logits and respects score ordering."""
        # Uniform logits → maximum score (most uncertain)
        uniform_logits = torch.zeros(1, 5)
        # Peaked logits → minimum score (most confident)
        peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])

        s_uniform = GEN.score(uniform_logits).item()
        s_peaked = GEN.score(peaked_logits).item()

        self.assertGreater(s_uniform, s_peaked)

    def test_predict_logits_batch(self):
        """Batch of logits produces correct shape."""
        logits = torch.randn(32, 10)
        scores = GEN(None).predict_logits(logits)
        self.assertEqual(scores.shape, (32,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_scores_are_non_negative(self):
        """GEN scores are always >= 0."""
        model = ClassificationModel()
        detector = GEN(model)
        x = torch.randn(32, 10)
        with torch.no_grad():
            scores = detector(x)
        self.assertTrue((scores >= 0).all())

    def test_gamma_parameter(self):
        """Different gamma values produce different scores, and gamma=1 relates to Gini impurity."""
        logits = torch.randn(8, 5)

        scores_01 = GEN.score(logits, gamma=0.1)
        scores_05 = GEN.score(logits, gamma=0.5)

        # Different gammas should produce different scores
        self.assertFalse(torch.allclose(scores_01, scores_05))

        # gamma=1 should give sum of p*(1-p), i.e. Gini impurity
        p = logits.softmax(dim=1)
        expected_gini = (p * (1 - p)).sum(dim=1)
        scores_1 = GEN.score(logits, gamma=1.0)
        self.assertTrue(torch.allclose(scores_1, expected_gini, atol=1e-6))

    def test_mock_performance(self):
        """
        Train a model on well-separated Gaussians. OOD inputs (far-away cluster)
        should yield higher GEN scores than ID inputs.
        Verifies AUROC > 0.90 on this simple synthetic problem.
        """
        torch.manual_seed(42)
        n_dim, n_classes, n_per_class = 10, 3, 100

        x_id, y_id, x_ood, y_ood = _make_id_ood_data(
            n_dim=n_dim, n_classes=n_classes, n_per_class=n_per_class
        )

        model = ClassificationModel(num_inputs=n_dim, num_outputs=n_classes, n_hidden=32)
        _train_model(model, x_id, y_id, epochs=300)

        detector = GEN(model)

        metrics = OODMetrics()
        with torch.no_grad():
            for x, y in DataLoader(TensorDataset(x_id, y_id), batch_size=64):
                metrics.update(detector(x), y)
            for x, y in DataLoader(TensorDataset(x_ood, y_ood), batch_size=64):
                metrics.update(detector(x), y)

        results = metrics.compute()
        self.assertGreater(
            results["AUROC"], 0.90, f"Expected AUROC > 0.90, got {results['AUROC']:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
