from unittest import mock

import pytest
import torch

from pytorch_ood.detector import GradNorm
from pytorch_ood.detector import gradnorm as gradnorm_module
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


class TestGradNormReference:
    def setup_method(self):
        torch.manual_seed(0)
        self.model = ClassificationModel(num_inputs=10, n_hidden=16, num_outputs=5).eval()
        # larger weights so that the predictions are far from uniform
        with torch.no_grad():
            self.model.classifier.weight.mul_(5)
        self.x = torch.randn(12, 10)

    def _detector(self, **kwargs):
        return GradNorm(
            self.model, param_filter=lambda name: name == "classifier.weight", **kwargs
        )

    @pytest.mark.parametrize("temperature", (1.0, 0.5, 10.0))
    def test_matches_reference(self, temperature):
        expected = _reference_scores(self.model, self.x, temperature)
        scores = self._detector(temperature=temperature, micro_batch_size=5)(self.x)
        torch.testing.assert_close(scores, expected)

    def test_sequential_path_matches_reference(self):
        with mock.patch.object(gradnorm_module, "_TORCH_FUNC_AVAILABLE", False):
            scores = self._detector(temperature=2.0)(self.x)
        torch.testing.assert_close(scores, _reference_scores(self.model, self.x, 2.0))
