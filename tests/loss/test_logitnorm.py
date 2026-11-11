import torch

from pytorch_ood.loss import LogitNorm

torch.manual_seed(123)


def test_matches_reference_formula():
    """
    Loss must equal cross-entropy on temperature-scaled, L2-normalized logits,
    as defined in the LogitNorm paper.
    """
    t = 0.04
    criterion = LogitNorm(t=t)
    logits = torch.randn(size=(32, 10))
    target = torch.randint(0, 10, size=(32,))

    loss = criterion(logits, target)

    norm = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
    expected = torch.nn.functional.cross_entropy(logits / (t * norm), target)

    assert torch.allclose(loss, expected)
