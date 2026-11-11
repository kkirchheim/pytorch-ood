import torch

from pytorch_ood.detector import GMM


def _classes(sizes=(100, 400, 300, 50), scales=(0.3, 1.0, 3.0, 0.5), dim=8, seed=0):
    """Gaussian classes with different sizes and spreads, so weights and determinants matter."""
    g = torch.Generator().manual_seed(seed)
    z = torch.cat(
        [
            torch.randn(n, dim, generator=g) * s + 4 * torch.eye(len(sizes), dim)[k]
            for k, (n, s) in enumerate(zip(sizes, scales))
        ]
    )
    y = torch.cat([torch.full((n,), k) for k, n in enumerate(sizes)])
    return z, y


def _mixture_nll(z_fit, y_fit, z, reg):
    """-log sum_k pi_k N(z | mu_k, Sigma_k), with torch.distributions as reference."""
    z_fit, z = z_fit.double(), z.double()
    log_probs = []
    for k in y_fit.unique():
        z_k = z_fit[y_fit == k]
        mu = z_k.mean(dim=0)
        cov = (z_k - mu).T @ (z_k - mu) / len(z_k) + reg * torch.eye(z.shape[1], dtype=z.dtype)
        normal = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        log_probs.append(normal.log_prob(z) + torch.tensor(len(z_k) / len(z_fit)).log())
    return -torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)


def test_matches_mixture_nll():
    torch.manual_seed(123)
    z, y = _classes()
    test = torch.randn(64, 8) * 2 + 1
    detector = GMM(None, reg=1e-3).fit_features(z, y)
    torch.testing.assert_close(
        detector.predict_features(test).double(), _mixture_nll(z, y, test, reg=1e-3)
    )


def test_fewer_samples_than_dimensions():
    """Singular class covariances are regularized and give finite scores."""
    torch.manual_seed(123)
    z, y = _classes(sizes=(5, 5), scales=(1.0, 1.0), dim=32)
    scores = GMM(None).fit_features(z, y).predict_features(torch.randn(10, 32))
    assert torch.isfinite(scores).all()
