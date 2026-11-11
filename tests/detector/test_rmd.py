import torch

from pytorch_ood.detector import RMD


def test_background_covariance_is_normalized():
    """The background Gaussian uses the covariance of all known features (divided by N)."""
    torch.manual_seed(0)
    y = torch.arange(40) % 2
    z = torch.randn(40, 10) + 3 * y.unsqueeze(1)
    detector = RMD(None).fit_features(z, y)
    centered = z.double() - z.double().mean(dim=0)
    expected = centered.T @ centered / len(z) + 1e-10 * torch.eye(10, dtype=torch.float64)
    torch.testing.assert_close(detector.background_cov, expected.float())
    torch.testing.assert_close(detector.background_precision, torch.linalg.inv(expected).float())


def test_scores_follow_fitted_parameters():
    """The score is the smallest difference between the squared distance to a class center and
    the squared distance to the background mean, under the fitted parameters."""
    g = torch.Generator().manual_seed(0)
    y = torch.arange(200) % 10
    z = torch.randn(200, 20, generator=g) + 2 * torch.randn(10, 20, generator=g)[y]
    x = torch.randn(32, 20, generator=g)
    detector = RMD(None).fit_features(z, y)

    x64, precision = x.double(), detector.precision.double()
    d_k = torch.stack(
        [((x64 - m) @ precision * (x64 - m)).sum(dim=1) for m in detector.mu.double()], dim=1
    )
    centered = x64 - detector.background_mu.double()
    d_0 = (centered @ detector.background_precision.double() * centered).sum(dim=1)
    expected = (d_k - d_0[:, None]).min(dim=1).values
    torch.testing.assert_close(
        detector.predict_features(x).double(), expected, rtol=1e-4, atol=1e-4
    )
