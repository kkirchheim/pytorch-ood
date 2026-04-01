import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import GradNormKL
from src.pytorch_ood.model import WideResNet
from tests.helpers import ClassificationModel


class TestGradNormKL(unittest.TestCase):
    def test_output_shape(self):
        model = WideResNet(num_classes=10).eval()
        detector = GradNormKL(model, param_filter=lambda x: x.startswith("fc."))

        x = torch.randn(size=(4, 3, 32, 32))
        output = detector(x)

        self.assertEqual(output.shape, (4,))

    def test_sign_convention(self):
        """OOD inputs (uniform softmax) should get higher scores than ID inputs (peaked softmax)."""
        model = WideResNet(num_classes=10).eval()
        detector = GradNormKL(model, param_filter=lambda x: x.startswith("fc."))

        # Near-zero input → model predicts near-uniform softmax (simulates OOD uncertainty)
        x_ood = torch.zeros(1, 3, 32, 32)
        # Large-magnitude input → model tends toward a peaked softmax (simulates ID confidence)
        torch.manual_seed(0)
        x_id = torch.randn(1, 3, 32, 32) * 10

        score_ood = detector(x_ood).item()
        score_id = detector(x_id).item()

        # In the OOD convention (higher = more OOD), OOD score should be >= ID score
        self.assertGreaterEqual(score_ood, score_id)

    def test_mock_dataset(self):
        """
        Runs GradNormKL on a small mock dataset (ClassificationModel + TensorDataset).

        The model is briefly trained on in-distribution data so that it becomes confident on ID
        inputs and less confident on OOD inputs. We then verify that:
        - output shape matches the dataset size
        - all scores are finite
        - mean OOD score is higher (more OOD) than mean ID score
        """
        torch.manual_seed(42)
        n_classes = 3
        n_dim = 10
        n_per_class = 30

        model = ClassificationModel(num_inputs=n_dim, num_outputs=n_classes)

        # Build a small training set: 3 well-separated Gaussian clusters
        centers = torch.tensor(
            [
                [5.0] + [0.0] * (n_dim - 1),
                [0.0, 5.0] + [0.0] * (n_dim - 2),
                [0.0, 0.0, 5.0] + [0.0] * (n_dim - 3),
            ]
        )
        x_train = torch.cat([torch.randn(n_per_class, n_dim) * 0.3 + c for c in centers])
        y_train = torch.repeat_interleave(torch.arange(n_classes), n_per_class)

        # Quick training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x_train), y_train).backward()
            optimizer.step()

        model.eval()
        detector = GradNormKL(model, param_filter=lambda name: name.startswith("classifier"))

        # ID data: same distribution as training
        x_id = torch.cat([torch.randn(10, n_dim) * 0.3 + c for c in centers])
        # OOD data: far from all training clusters
        x_ood = torch.randn(30, n_dim) * 0.3 + 20.0

        dataset = TensorDataset(torch.cat([x_id, x_ood]))
        loader = DataLoader(dataset, batch_size=15)

        all_scores = []
        for (batch,) in loader:
            all_scores.append(detector(batch))
        scores = torch.cat(all_scores)

        # Shape check
        self.assertEqual(scores.shape, (60,))

        # All scores should be finite
        self.assertTrue(torch.isfinite(scores).all())

        # OOD scores should be higher on average (less confident → smaller gradient norm →
        # less negative negated score)
        mean_id_score = scores[:30].mean().item()
        mean_ood_score = scores[30:].mean().item()
        self.assertGreater(mean_ood_score, mean_id_score)
