import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import Mahalanobis, MultiMahalanobis


def _random_gaussians(n, d, n_classes, g):
    """Classes with distinct means and a random (correlated) shared covariance."""
    y = torch.arange(n) % n_classes
    mix = torch.randn(d, d, generator=g)
    return torch.randn(n, d, generator=g) @ mix + 3 * torch.randn(n_classes, d, generator=g)[y], y


@pytest.mark.parametrize("kwargs", [{}, {"eps": 1e-2}, {"use_pinv": True}], ids=str)
def test_layers_score_like_independent_mahalanobis_detectors(kwargs):
    """Each layer uses its own Gaussians: the score is the alpha-weighted sum of the scores of one
    Mahalanobis detector per layer."""
    g = torch.Generator().manual_seed(0)
    layers = [_random_gaussians(90, 6, 3, g) for _ in range(3)]
    y = layers[0][1]
    zs = [z for z, _ in layers]
    xs = [torch.randn(16, 6, generator=g) @ torch.randn(6, 6, generator=g) for _ in range(3)]
    alpha = [1.0, 2.0, 3.0]
    detector = MultiMahalanobis([torch.nn.Identity()] * 3, alpha=alpha, **kwargs).fit_structured(
        zs, y
    )
    expected = sum(
        a * Mahalanobis(None, **kwargs).fit_features(z, y).predict_features(x)
        for a, z, x in zip(alpha, zs, xs)
    )
    torch.testing.assert_close(detector.predict_structured(xs), expected)


def test_feature_maps_are_averaged_spatially():
    """Per-layer feature maps of shape B x C x H x W score like their spatial means, both in
    ``predict_structured`` and in ``predict``, where each layer gets the previous layer's output."""
    torch.manual_seed(0)
    layers = [torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 5, 3)]
    y = torch.arange(30) % 3
    images = torch.randn(30, 3, 8, 8) + torch.randn(3, 3, 1, 1)[y]
    detector = MultiMahalanobis(layers).fit(DataLoader(TensorDataset(images, y), batch_size=8))

    x = torch.randn(5, 3, 8, 8)
    with torch.no_grad():
        maps = [layers[0](x)]
        maps.append(layers[1](maps[0]))
    means = [m.mean(dim=(2, 3)) for m in maps]
    expected = detector.predict_structured(means)
    torch.testing.assert_close(detector.predict_structured(maps), expected)
    torch.testing.assert_close(detector.predict(x), expected)


def test_fit_from_a_shuffling_loader_pairs_features_with_their_labels():
    """The features of every layer are paired with the labels of the same samples, so the order in
    which a loader yields them does not matter: a shuffling loader that gives each pass over it
    another order fits the same Gaussians as ``fit_structured`` on the spatial means of the layers
    in dataset order. Each batch passes through the layers once."""
    torch.manual_seed(0)
    layers = [torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 5, 3), torch.nn.Conv2d(5, 6, 3)]
    y = torch.arange(60) % 3
    images = torch.randn(60, 3, 10, 10) + 2 * torch.randn(3, 3, 1, 1)[y]
    data = TensorDataset(images, y)

    with torch.no_grad():
        maps = [layers[0](images)]
        maps.append(layers[1](maps[0]))
        maps.append(layers[2](maps[1]))
    expected = MultiMahalanobis(layers).fit_structured([m.mean(dim=(2, 3)) for m in maps], y)

    shuffling = DataLoader(
        data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1)
    )
    forward_passes = []
    handle = layers[0].register_forward_hook(lambda *_: forward_passes.append(1))
    detector = MultiMahalanobis(layers).fit(shuffling)
    handle.remove()

    # summing in another order changes the statistics only by rounding errors
    torch.testing.assert_close(detector.mu, expected.mu, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(detector.cov, expected.cov, rtol=1e-4, atol=1e-5)
    assert len(forward_passes) == len(shuffling)
