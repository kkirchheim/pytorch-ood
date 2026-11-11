import torch

from pytorch_ood.detector import ViM


def test_ill_conditioned_features_match_float64_reference():
    torch.manual_seed(0)
    n, dim, c, d = 2000, 64, 4, 16
    scales = torch.logspace(0, -6, dim)
    rotation = torch.linalg.qr(torch.randn(dim, dim))[0]
    # ReLU features with a large mean, as penultimate CNN features: ViM centers at u, not at the mean
    mean = 5.0 * torch.rand(dim)
    features = ((torch.randn(n, dim) * scales) @ rotation.T + mean).relu()
    x = ((torch.randn(32, dim) * scales) @ rotation.T + 1.2 * mean).relu()
    w, b = torch.randn(c, dim), torch.randn(c)

    detector = ViM(None, d=d, w=w, b=b).fit_features(features, torch.zeros(n).long())
    residual = ((x - detector.u) @ detector.residual_subspace).norm(dim=1)

    u = -(torch.linalg.pinv(w.double()) @ b.double())
    centered = features.double() - u
    _, vecs = torch.linalg.eigh(centered.T @ centered / n)
    expected = ((x.double() - u) @ vecs[:, : dim - d]).norm(dim=1)

    torch.testing.assert_close(residual.double(), expected, rtol=1e-3, atol=1e-6)
