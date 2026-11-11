import unittest

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.detector import Mahalanobis, MultiMahalanobis
from tests.helpers import points_with_covariance
from tests.helpers.model import ConvClassifier


class MultiMahalanobisTest(unittest.TestCase):
    """
    Test multi-layer mahalanobis method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_refit_does_not_accumulate(self):
        """Re-fitting replaces parameters instead of appending (e.g. during HPO)."""
        nn = ConvClassifier(in_channels=3, out_channels=16)
        layers = [nn.layer1, nn.pool]
        model = MultiMahalanobis(layers)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 3, 16, 16))
        loader = DataLoader(TensorDataset(x, y))

        model.fit(loader)
        first_mu = [m.clone() for m in model.mu]

        model.fit(loader)

        # one set of parameters per layer, not accumulated across fits
        self.assertEqual(len(model.mu), len(layers))
        self.assertEqual(len(model.cov), len(layers))
        self.assertEqual(len(model.precision), len(layers))
        # same data => same fitted centers
        for a, b in zip(first_mu, model.mu):
            self.assertTrue(torch.allclose(a, b))

        scores = model(x)
        self.assertEqual(scores.shape[0], 20)


class MultiMahalanobisScoreTest(unittest.TestCase):
    def test_weighted_sum_of_half_squared_distances_to_closest_center(self):
        detector = MultiMahalanobis(
            [torch.nn.Identity(), torch.nn.Identity()], alpha=[1.0, 2.0], eps=0.0
        )
        # layer 1: centers (0, 0), (4, 0), precision diag(1, 4); layer 2: (0, 0), (0, 2), identity
        z1, y = points_with_covariance(
            torch.tensor([[0.0, 0.0], [4.0, 0.0]]), torch.tensor([1.0, 0.25])
        )
        z2, _ = points_with_covariance(torch.tensor([[0.0, 0.0], [0.0, 2.0]]), torch.ones(2))
        detector.fit_structured([z1, z2], y)
        torch.testing.assert_close(detector.precision[0], torch.diag(torch.tensor([1.0, 4.0])))
        torch.testing.assert_close(detector.precision[1], torch.eye(2))
        x = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
        # layer 1: squared distances (5, 13) and (9, 1); layer 2: (2, 2) and (9, 13)
        expected = 1.0 * torch.tensor([2.5, 0.5]) + 2.0 * torch.tensor([1.0, 4.5])
        torch.testing.assert_close(detector.predict_structured([x, x]), expected)


class MultiMahalanobisSingleLayerTest(unittest.TestCase):
    def test_single_layer_equals_mahalanobis(self):
        """One identity layer with weight 1 fits the same Gaussian as Mahalanobis."""
        torch.manual_seed(0)
        y = torch.arange(60) % 3
        z = torch.randn(60, 80) + torch.randn(3, 80)[y]  # fewer samples than dimensions
        x = torch.randn(16, 80)
        for kwargs in [{}, {"eps": 1e-2}, {"use_pinv": True}]:
            with self.subTest(**kwargs):
                multi = MultiMahalanobis([torch.nn.Identity()], **kwargs).fit_structured([z], y)
                single = Mahalanobis(None, **kwargs).fit_features(z, y)
                torch.testing.assert_close(multi.precision[0], single.precision)
                torch.testing.assert_close(
                    multi.predict_structured([x]), single.predict_features(x)
                )


def _random_gaussians(n, d, n_classes, g):
    """Classes with distinct means and a random (correlated) shared covariance."""
    y = torch.arange(n) % n_classes
    mix = torch.randn(d, d, generator=g)
    return torch.randn(n, d, generator=g) @ mix + 3 * torch.randn(n_classes, d, generator=g)[y], y


def test_layers_score_like_independent_mahalanobis_detectors():
    """Each layer uses its own Gaussians: the score is the alpha-weighted sum of the scores of one
    Mahalanobis detector per layer."""
    g = torch.Generator().manual_seed(0)
    layers = [_random_gaussians(90, 6, 3, g) for _ in range(3)]
    y = layers[0][1]
    zs = [z for z, _ in layers]
    xs = [torch.randn(16, 6, generator=g) @ torch.randn(6, 6, generator=g) for _ in range(3)]
    alpha = [1.0, 2.0, 3.0]
    detector = MultiMahalanobis([torch.nn.Identity()] * 3, alpha=alpha).fit_structured(zs, y)
    expected = sum(
        a * Mahalanobis(None).fit_features(z, y).predict_features(x)
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


def test_parameters_set_by_hand_require_fit():
    detector = MultiMahalanobis([torch.nn.Identity()])
    detector.mu = [torch.zeros(2, 3)]
    detector.precision = [torch.eye(3)]
    with pytest.raises(RequiresFittingException):
        detector.predict_structured([torch.randn(4, 3)])
