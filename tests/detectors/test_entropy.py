import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import Entropy
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


class TestEntropy(unittest.TestCase):
    """Tests for the Entropy OOD detector."""

    def test_output_shape_classification(self):
        """Scalar score per sample for 1-D classification inputs."""
        model = ClassificationModel(num_inputs=10, num_outputs=5)
        detector = Entropy(model)
        x = torch.randn(16, 10)
        with torch.no_grad():
            scores = detector(x)
        self.assertEqual(scores.shape, (16,))

    def test_output_shape_segmentation(self):
        """Pixel-wise score for segmentation model (B, H, W) output."""
        model = SegmentationModel(in_channels=3, out_channels=4)
        detector = Entropy(model)
        x = torch.randn(4, 3, 8, 8)
        with torch.no_grad():
            scores = detector(x)
        self.assertEqual(scores.shape, (4, 8, 8))

    def test_predict_logits(self):
        """predict_logits works on raw logits and respects entropy ordering."""
        # Uniform logits → maximum entropy
        uniform_logits = torch.zeros(1, 5)
        # Peaked logits → minimum entropy
        peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])

        s_uniform = Entropy.score(uniform_logits).item()
        s_peaked = Entropy.score(peaked_logits).item()

        self.assertGreater(s_uniform, s_peaked)

        # Batch of logits produces correct shape
        logits = torch.randn(32, 10)
        scores = Entropy(None).predict_logits(logits)
        self.assertEqual(scores.shape, (32,))

    def test_scores_are_non_negative(self):
        """Entropy is always >= 0."""
        model = ClassificationModel()
        detector = Entropy(model)
        x = torch.randn(32, 10)
        with torch.no_grad():
            scores = detector(x)
        self.assertTrue((scores >= 0).all())

    def test_mock_performance(self):
        """
        Train a model on well-separated Gaussians. OOD inputs (far-away cluster)
        should yield higher entropy (more uncertain) than ID inputs on average.
        Verifies AUROC > 0.90 on this simple synthetic problem.
        """
        torch.manual_seed(42)
        n_dim, n_classes, n_per_class = 10, 3, 100

        x_id, y_id, x_ood, y_ood = _make_id_ood_data(
            n_dim=n_dim, n_classes=n_classes, n_per_class=n_per_class
        )

        model = ClassificationModel(num_inputs=n_dim, num_outputs=n_classes, n_hidden=32)
        _train_model(model, x_id, y_id, epochs=300)

        detector = Entropy(model)

        metrics = OODMetrics()
        with torch.no_grad():
            loader = DataLoader(TensorDataset(x_id, y_id), batch_size=64)
            for x, y in loader:
                metrics.update(detector(x), y)

            loader_ood = DataLoader(TensorDataset(x_ood, y_ood), batch_size=64)
            for x, y in loader_ood:
                metrics.update(detector(x), y)

        results = metrics.compute()
        self.assertGreater(
            results["AUROC"], 0.90, f"Expected AUROC > 0.90, got {results['AUROC']:.4f}"
        )
