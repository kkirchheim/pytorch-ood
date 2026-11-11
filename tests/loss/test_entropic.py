import pytest
import torch
import torch.nn.functional as F

from pytorch_ood.loss import EntropicOpenSetLoss


@pytest.mark.parametrize("n_ood", [0, 6], ids=["all-id", "mixed"])
def test_cross_entropy_on_id_rows_and_uniform_target_on_ood_rows(
    n_ood: int, seed: torch.Generator
):
    logits = torch.randn(12, 4, generator=seed)
    target = torch.arange(12) % 4
    target[:n_ood] = -1
    known = target >= 0
    expected = torch.zeros(12)
    expected[known] = F.cross_entropy(logits[known], target[known], reduction="none")
    expected[~known] = -logits[~known].log_softmax(dim=1).mean(dim=1)

    torch.testing.assert_close(EntropicOpenSetLoss(reduction="none")(logits, target), expected)
