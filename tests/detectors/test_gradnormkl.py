import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import GradNormKL
from src.pytorch_ood.detector import gradnormkl as gradnormkl_module
from src.pytorch_ood.model import WideResNet
from tests.helpers import ClassificationModel


def _reference_scores(model, x, temperature=1.0):
    # the paper, written out per sample: D_KL(u || softmax(f(x) / T)) up to a constant, and the
    # l1-norm of its gradient w.r.t. the weights of the last layer
    scores = []
    for xi in x:
        weight = model.classifier.weight
        logits = model(xi.unsqueeze(0)) / temperature
        num_classes = logits.shape[1]
        loss = -(logits.log_softmax(dim=1) / num_classes).sum()
        (grad,) = torch.autograd.grad(loss, weight)
        scores.append(grad.abs().sum())
    return -torch.stack(scores)


class TestGradNormKLReference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ClassificationModel(num_inputs=10, n_hidden=16, num_outputs=5).eval()
        # larger weights so that the predictions are far from uniform
        with torch.no_grad():
            self.model.classifier.weight.mul_(5)
        self.x = torch.randn(12, 10)

    def _detector(self, **kwargs):
        return GradNormKL(
            self.model, param_filter=lambda name: name == "classifier.weight", **kwargs
        )

    def test_matches_reference(self):
        for temperature in (1.0, 0.5, 10.0):
            with self.subTest(temperature=temperature):
                expected = _reference_scores(self.model, self.x, temperature)
                scores = self._detector(temperature=temperature, micro_batch_size=5)(self.x)
                torch.testing.assert_close(scores, expected)

    def test_sequential_path_matches_reference(self):
        with mock.patch.object(gradnormkl_module, "_TORCH_FUNC_AVAILABLE", False):
            scores = self._detector(temperature=2.0)(self.x)
        torch.testing.assert_close(scores, _reference_scores(self.model, self.x, 2.0))

    def test_logit_gradient(self):
        # the gradient w.r.t. the logits is (softmax(z / T) - 1/C) / T
        logits = torch.randn(1, 5, requires_grad=True)
        loss = gradnormkl_module._uniform_cross_entropy(logits, 2.0)
        (grad,) = torch.autograd.grad(loss, logits)
        expected = ((logits / 2.0).softmax(dim=1) - 1 / 5) / 2.0
        torch.testing.assert_close(grad, expected.detach())

    def test_default_temperature(self):
        self.assertEqual(GradNormKL(self.model).temperature, 1.0)


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
