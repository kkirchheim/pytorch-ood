import torch

from pytorch_ood.detector import GEN


def test_gamma_parameter():
    """Different gamma values produce different scores, and gamma=1 relates to Gini impurity."""
    logits = torch.randn(8, 5)

    scores_01 = GEN.score(logits, gamma=0.1)
    scores_05 = GEN.score(logits, gamma=0.5)

    # Different gammas should produce different scores
    assert not torch.allclose(scores_01, scores_05)

    # gamma=1 should give mean of p*(1-p), i.e. average Gini impurity
    p = logits.softmax(dim=1)
    expected_gini = (p * (1 - p)).mean(dim=1)
    scores_1 = GEN.score(logits, gamma=1.0)
    assert torch.allclose(scores_1, expected_gini, atol=1e-06)


def test_M_none_matches_full_class_count():
    """M=None (all classes) equals explicitly summing over all C classes."""
    logits = torch.randn(8, 5)
    assert torch.allclose(GEN.score(logits, M=None), GEN.score(logits, M=5))


def test_M_truncates_to_top_probabilities():
    """M < C sums the power transform over only the M largest probabilities."""
    logits = torch.randn(8, 10)
    gamma, M = 0.1, 3

    scores = GEN.score(logits, gamma=gamma, M=M)

    p = logits.softmax(dim=1).clamp(1e-7, 1 - 1e-7)
    top = p.sort(dim=1, descending=True).values[:, :M]
    expected = (top.pow(gamma) * (1 - top).pow(gamma)).mean(dim=1)

    assert torch.allclose(scores, expected, atol=1e-06)
    # truncation changes the score relative to using all classes
    assert not torch.allclose(scores, GEN.score(logits, gamma=gamma))


def test_M_via_constructor():
    """The constructor's M is forwarded to predict_logits."""
    logits = torch.randn(8, 10)
    detector = GEN(None, gamma=0.1, M=5)
    assert torch.allclose(detector.predict_logits(logits), GEN.score(logits, M=5))
