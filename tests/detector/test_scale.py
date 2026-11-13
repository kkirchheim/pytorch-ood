import numpy as np
import torch

from pytorch_ood.detector import SCALE
from pytorch_ood.detector.ash import ash_s
from pytorch_ood.detector.scale import scale


def test_does_not_prune():
    """
    SCALE keeps all activations (only scales), whereas ASH-S prunes the
    smallest ``percentile`` fraction to zero. So SCALE must have no extra
    zeros beyond those already present in the input.
    """
    x = torch.rand(size=(4, 8, 5, 5)) + 0.1  # strictly positive
    scale_out = scale(x.clone(), percentile=0.65)
    ash_out = ash_s(x.clone(), percentile=0.65)

    # SCALE leaves every position non-zero, ASH-S zeros most of them
    assert (scale_out == 0).sum().item() == 0
    assert (ash_out == 0).sum().item() > 0


def test_scale_factor_positive():
    """Scaling factor exp(s1/s2) >= 1, so output magnitude grows."""
    x = torch.rand(size=(4, 8, 5, 5)) + 0.1
    out = scale(x.clone(), percentile=0.65)
    # since s1 >= s2 > 0 for non-negative activations, scale factor >= 1
    assert torch.all(out.abs() >= x.abs() - 1e-06)


def test_separates_gaussians():
    """
    Synthetic sanity check: ID and OOD features with different norms should
    be separable by the resulting energy score.
    """
    from pytorch_ood.metrics import OODMetrics

    torch.manual_seed(0)
    head = torch.nn.Linear(16, 10)
    detector = SCALE(backbone=None, head=lambda f: head(f.flatten(1)), percentile=0.65)

    # ID: large-norm features, OOD: small-norm features
    z_in = torch.randn(128, 16, 1, 1).abs() * 3.0
    z_out = torch.randn(128, 16, 1, 1).abs() * 0.5

    s_in = detector.predict_feature_maps(z_in)
    s_out = detector.predict_feature_maps(z_out)

    metrics = OODMetrics()
    metrics.update(s_in, torch.zeros(128))  # ID labels >= 0
    metrics.update(s_out, -torch.ones(128).long())  # OOD labels < 0
    auroc = metrics.compute()["AUROC"]
    assert auroc > 0.6


def test_all_zero_feature_map_is_not_scaled():
    # s1 = s2 = 0 gives 0 / 0 (the official code returns NaN); the decided ratio is 0, i.e. factor 1
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]).reshape(2, 4, 1, 1)

    # n = 4, percentile 0.5 gives k = 2, so s1 = 10 and s2 = 4 + 3 = 7 for the first map
    expected = torch.cat([x[:1] * np.exp(10 / 7), torch.zeros(1, 4, 1, 1)]).float()
    torch.testing.assert_close(scale(x, 0.5), expected)

    detector = SCALE(backbone=None, head=lambda f: f.flatten(1), percentile=0.5)
    alone = detector.predict_feature_maps(x[:1])
    mixed = detector.predict_feature_maps(x)
    assert torch.isfinite(mixed).all()
    assert torch.equal(mixed[:1], alone)
    # the zero map gives four zero logits, whose energy is -log(4)
    torch.testing.assert_close(mixed[1], torch.tensor(-np.log(4.0)).float())
