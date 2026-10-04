import unittest
from unittest import mock

import torch

from src.pytorch_ood.detector import GradUncertainty
from src.pytorch_ood.detector import graduncertainty as graduncertainty_module
from src.pytorch_ood.model import WideResNet


def _linear_head(bias: torch.Tensor) -> torch.nn.Linear:
    """Linear layer with zero weights, so that the softmax is determined by ``bias``."""
    head = torch.nn.Linear(16, len(bias))
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(bias)
    return head


class TestGradUncertainty(unittest.TestCase):
    """
    Tests for GradUncertainty
    """

    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = GradUncertainty(model, param_filter=lambda x: x.startswith("fc."))

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (16,))

    def test_uniform_output_scores_higher_than_confident_output(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16)
        uniform = _linear_head(torch.zeros(10))
        confident = _linear_head(torch.tensor([10.0] + [0.0] * 9))
        for batched in (True, False):
            with self.subTest(batched=batched):
                with mock.patch.object(graduncertainty_module, "_TORCH_FUNC_AVAILABLE", batched):
                    score_uniform = GradUncertainty(uniform)(x)
                    score_confident = GradUncertainty(confident)(x)
                self.assertTrue((score_uniform > score_confident).all())

    def test_matches_closed_form(self):
        # for a linear layer, the loss gradient w.r.t. the logits is C * p - 1, so the squared
        # gradient norms are ||C * p - 1||^2 * ||x||^2 (weight) and ||C * p - 1||^2 (bias)
        torch.manual_seed(0)
        x = torch.randn(4, 16)
        head = torch.nn.Linear(16, 5)
        p = head(x).softmax(dim=1)
        expected = -((5 * p - 1) ** 2).sum(dim=1) * ((x**2).sum(dim=1) + 1)
        for batched in (True, False):
            with self.subTest(batched=batched):
                with mock.patch.object(graduncertainty_module, "_TORCH_FUNC_AVAILABLE", batched):
                    scores = GradUncertainty(head)(x)
                torch.testing.assert_close(scores, expected.detach())
