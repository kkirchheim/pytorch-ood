import math
import unittest

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import Entropy
from pytorch_ood.metrics import OODMetrics
from tests.helpers import ClassificationModel


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


@pytest.mark.parametrize("margin", [0.0, 5.0, 20.0, 30.0, 40.0])
def test_matches_closed_form(margin: float):
    # logits [m, 0, 0]: with e = exp(-m), the entropy is log(1 + 2e) + 2em / (1 + 2e)
    e = math.exp(-margin)
    expected = math.log1p(2 * e) + 2 * e * margin / (1 + 2 * e)
    scores = Entropy(None).predict_logits(torch.tensor([[margin, 0.0, 0.0]]))
    # float32 rounds the largest softmax value to 1, which loses about 1 / (m + 1) of the entropy
    torch.testing.assert_close(scores, torch.tensor([expected]), rtol=0.1, atol=0)


class TestEntropy(unittest.TestCase):
    """Tests for the Entropy OOD detector."""

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
