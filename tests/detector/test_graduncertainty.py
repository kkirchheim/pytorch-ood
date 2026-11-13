from unittest import mock

import torch

from pytorch_ood.detector import GradUncertainty
from pytorch_ood.detector import graduncertainty as graduncertainty_module


def _linear_head(bias: torch.Tensor) -> torch.nn.Linear:
    """Linear layer with zero weights, so that the softmax is determined by ``bias``."""
    head = torch.nn.Linear(16, len(bias))
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(bias)
    return head.eval()


def test_matches_closed_form():
    # for a linear layer, the loss gradient w.r.t. the logits is C * p - 1, so the squared
    # gradient norms are ||C * p - 1||^2 * ||x||^2 (weight) and ||C * p - 1||^2 (bias)
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    heads = {
        "random": torch.nn.Linear(16, 5).eval(),
        "uniform": _linear_head(torch.zeros(5)),
        "margin 20": _linear_head(torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0])),
        # the other softmax values are below the 1e-12 clamp of binary cross-entropy
        "margin 40": _linear_head(torch.tensor([40.0, 0.0, 0.0, 0.0, 0.0])),
    }
    for name, head in heads.items():
        p = head(x).softmax(dim=1)
        expected = -((5 * p - 1) ** 2).sum(dim=1) * ((x**2).sum(dim=1) + 1)
        for batched in (True, False):
            with mock.patch.object(graduncertainty_module, "_TORCH_FUNC_AVAILABLE", batched):
                scores = GradUncertainty(head)(x)
            torch.testing.assert_close(scores, expected.detach())
