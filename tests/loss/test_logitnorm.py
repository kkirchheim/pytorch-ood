import torch

from pytorch_ood.loss import LogitNorm

torch.manual_seed(123)


def test_loss_is_non_negative():
    """
    Cross-entropy is always non-negative. The previous implementation applied
    nll_loss to normalized logits (instead of log-probabilities), which
    produced negative losses for confident predictions.
    """
    criterion = LogitNorm()
    logits = torch.tensor([[10.0, -5.0, -5.0]])
    target = torch.tensor([0])

    loss = criterion(logits, target)

    assert loss.item() >= 0.0


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


def test_scale_invariance():
    """
    Scaling the logits by a constant factor must not change the loss;
    this is the core property of LogitNorm.
    """
    criterion = LogitNorm(t=1.0)
    logits = torch.randn(size=(16, 10))
    target = torch.randint(0, 10, size=(16,))

    loss_1 = criterion(logits, target)
    loss_2 = criterion(100.0 * logits, target)

    assert torch.allclose(loss_1, loss_2, atol=1e-05)
