import unittest

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.detector import Mahalanobis, MahalanobisODIN, MultiMahalanobis
from tests.helpers import ClassificationModel, points_with_covariance


class MahalanobisTest(unittest.TestCase):
    """
    Test mahalanobis method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_something(self):
        nn = ClassificationModel()
        model = Mahalanobis(nn)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(scores)

        scores = model(torch.ones(size=(10, 10)) * 10)
        print(scores)

        self.assertIsNotNone(scores)


class MahalanobisScoreTest(unittest.TestCase):
    def test_half_squared_distance_to_closest_center(self):
        centers = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
        z, y = points_with_covariance(centers, torch.tensor([1.0, 0.25]))
        detector = Mahalanobis(None, eps=0.0).fit_features(z, y)
        torch.testing.assert_close(detector.mu, centers)
        torch.testing.assert_close(detector.precision, torch.diag(torch.tensor([1.0, 4.0])))
        x = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
        # squared distances: (5, 13) and (9, 1)
        torch.testing.assert_close(detector.predict_features(x), torch.tensor([2.5, 0.5]))


class MahalanobisEpsTest(unittest.TestCase):
    """
    Tests for the configurable eps parameter added to the Mahalanobis detector.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_default_eps(self):
        """Mahalanobis should use eps=1e-10 by default."""
        model = Mahalanobis(None)
        self.assertEqual(model.eps, 1e-10)

    def test_custom_eps(self):
        """Mahalanobis should accept a custom eps value."""
        model = Mahalanobis(None, eps=1e-3)
        self.assertEqual(model.eps, 1e-3)

    def test_eps_applied_to_covariance(self):
        """The eps value should be added to the diagonal of the covariance matrix during fit."""
        x = torch.randn(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))]).long()

        model_a = Mahalanobis(None, eps=0.0).fit_features(x, y)
        model_b = Mahalanobis(None, eps=1e-2).fit_features(x, y)

        torch.testing.assert_close(model_b.cov - model_a.cov, torch.eye(10) * 1e-2)
        expected = torch.linalg.inv(model_b.cov.double()).float()
        torch.testing.assert_close(model_b.precision, expected)

    def test_eps_makes_singular_covariance_invertible(self):
        """
        eps=0 on near-singular data can lead to singular matrices.
        With eps > 0 the same data should produce a finite precision matrix.
        """
        nn = ClassificationModel()
        # Use identical samples to create a (near-)singular covariance
        x_const = torch.ones(size=(20, 10))
        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        dataset = TensorDataset(x_const, y)
        loader = DataLoader(dataset)

        model_safe = Mahalanobis(nn, eps=1e-3)
        model_safe.fit(loader)
        self.assertTrue(
            torch.isfinite(model_safe.precision).all(),
            "With eps=1e-3, precision matrix should be finite even for constant inputs.",
        )


class MahalanobisPinvTest(unittest.TestCase):
    """
    Tests for the use_pinv parameter of the Mahalanobis detector.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(size=(40, 10))
        self.y = torch.cat([torch.zeros(size=(20,)), torch.ones(size=(20,))]).long()

    def test_default_is_inv(self):
        self.assertFalse(Mahalanobis(None).use_pinv)

    def test_pinv_precision(self):
        model = Mahalanobis(None, use_pinv=True).fit_features(self.x, self.y)
        expected = torch.linalg.pinv(model.cov.double(), hermitian=True).float()
        torch.testing.assert_close(model.precision, expected)

    def test_pinv_ignores_eps(self):
        """With pinv, no ridge is added: directions without variance are discarded instead."""
        model_a = Mahalanobis(None, eps=0.0, use_pinv=True).fit_features(self.x, self.y)
        model_b = Mahalanobis(None, eps=1e-2, use_pinv=True).fit_features(self.x, self.y)
        torch.testing.assert_close(model_a.cov, model_b.cov)
        torch.testing.assert_close(model_a.precision, model_b.precision)

    def test_pinv_matches_inv(self):
        """For a well-conditioned covariance matrix, inv and pinv should give the same scores."""
        model_inv = Mahalanobis(None).fit_features(self.x, self.y)
        model_pinv = Mahalanobis(None, use_pinv=True).fit_features(self.x, self.y)
        x = torch.randn(size=(16, 10))
        torch.testing.assert_close(
            model_pinv.predict_features(x), model_inv.predict_features(x), rtol=1e-4, atol=1e-4
        )


def _reference_scores(z, y, x, eps, use_pinv):
    """Float64 reference: half the squared distance to the closest center under the shared
    within-class covariance (sum of centered outer products / N), like sklearn's EmpiricalCovariance."""
    z, x = z.double(), x.double()
    n_classes = int(y.max()) + 1
    mu = torch.stack([z[y == k].mean(dim=0) for k in range(n_classes)])
    centered = z - mu[y]
    cov = centered.T @ centered / len(z)
    if use_pinv:
        precision = torch.linalg.pinv(cov, hermitian=True)
    else:
        precision = torch.linalg.inv(cov + eps * torch.eye(z.shape[1], dtype=torch.float64))
    d = torch.stack([((x - m) @ precision * (x - m)).sum(dim=1) for m in mu], dim=1)
    return 0.5 * d.min(dim=1).values


class MahalanobisNumericsTest(unittest.TestCase):
    """
    Rank-deficient covariance (fewer samples than feature dimensions): float32 inputs must give
    the scores of a float64 computation.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        n, d = 60, 80
        # decaying feature scales make the covariance ill-conditioned, like penultimate CNN features
        scales = torch.logspace(0, -3, d)
        self.y = torch.arange(n) % 3
        self.z = (torch.randn(n, d) * scales + torch.randn(3, d)[self.y]).relu()
        self.x = (torch.randn(16, d) * scales + torch.randn(3, d)[torch.arange(16) % 3]).relu()

    def _check(self, **kwargs):
        detector = Mahalanobis(None, **kwargs).fit_features(self.z, self.y)
        scores = detector.predict_features(self.x)
        self.assertEqual(scores.dtype, torch.float32)
        expected = _reference_scores(
            self.z, self.y, self.x, eps=detector.eps, use_pinv=detector.use_pinv
        )
        torch.testing.assert_close(scores.double(), expected, rtol=1e-3, atol=1e-3)

    def test_inv_matches_float64_reference(self):
        self._check()

    def test_pinv_matches_float64_reference(self):
        self._check(use_pinv=True)

    def test_float64_input_stays_float64(self):
        detector = Mahalanobis(None).fit_features(self.z.double(), self.y)
        self.assertEqual(detector.mu.dtype, torch.float64)
        self.assertEqual(detector.precision.dtype, torch.float64)
        self.assertEqual(detector.predict_features(self.x.double()).dtype, torch.float64)

    def test_float32_input_gives_float32_parameters(self):
        detector = Mahalanobis(None).fit_features(self.z, self.y)
        self.assertEqual(detector.mu.dtype, torch.float32)
        self.assertEqual(detector.precision.dtype, torch.float32)


def _features_with_dead_units(n: int, d: int = 50, n_classes: int = 10, n_dead: int = 5):
    """Post-ReLU-like features of ``n_classes`` classes in which ``n_dead`` units never fire, so
    that the covariance has exact zero-variance directions (and is rank-deficient for ``n < d``)."""
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n) % n_classes
    z = (torch.randn(n, d, generator=g) + 2 * torch.randn(n_classes, d, generator=g)[y]).relu()
    z[:, :n_dead] = 0
    x = (torch.randn(32, d, generator=g) + torch.randn(32, d, generator=g)).relu()
    return z, y, x


def _half_squared_distances(x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor):
    """Float64 loop over classes: :math:`B \\times C` half squared Mahalanobis distances."""
    x, mu, precision = x.double(), mu.double(), precision.double()
    return 0.5 * torch.stack([((x - m) @ precision * (x - m)).sum(dim=1) for m in mu], dim=1)


@pytest.mark.parametrize("use_pinv", [False, True], ids=["inv", "pinv"])
@pytest.mark.parametrize("n", [200, 40], ids=["n>d", "n<d"])
def test_scores_follow_fitted_mean_and_precision(use_pinv, n):
    """The score is half the squared distance to the closest center under the fitted ``mu`` and
    ``precision``, also with zero-variance units and fewer samples than dimensions."""
    z, y, x = _features_with_dead_units(n)
    detector = Mahalanobis(None, use_pinv=use_pinv).fit_features(z, y)
    expected = _half_squared_distances(x, detector.mu, detector.precision).min(dim=1).values
    torch.testing.assert_close(
        detector.predict_features(x).double(), expected, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("norm_std", [None, [0.5, 2.0]], ids=["no-norm-std", "norm-std"])
def test_mahalanobis_odin_steps_against_gradient_of_closest_class_distance(norm_std):
    """MahalanobisODIN moves the input by ``eps`` against the sign of the gradient of the half
    squared distance to the closest center, with the step of channel ``i`` divided by
    ``norm_std[i]``, then scores it with Mahalanobis."""
    torch.manual_seed(0)
    encoder = torch.nn.Linear(6, 4)
    y = torch.arange(60) % 3
    with torch.no_grad():
        z = encoder(torch.randn(60, 6) + 2 * torch.randn(3, 6)[y])
    detector = MahalanobisODIN(encoder, eps=0.01, norm_std=norm_std).fit_features(z, y)
    reference = Mahalanobis(encoder).fit_features(z, y)

    x = torch.randn(8, 6, requires_grad=True)
    f = encoder(x)
    closest = _half_squared_distances(f.detach(), reference.mu, reference.precision).argmin(dim=1)
    centered = f - reference.mu[closest]
    (0.5 * ((centered @ reference.precision) * centered).sum(dim=1)).mean().backward()
    step = x.grad.sign()
    for i, std in enumerate(norm_std or []):
        step[:, i] /= std
    expected = reference.predict_features(encoder(x.detach() - 0.01 * step))

    torch.testing.assert_close(detector.predict(x.detach()), expected)


@pytest.mark.parametrize("multi", [False, True], ids=["Mahalanobis", "MultiMahalanobis"])
def test_near_singular_covariance_matches_float64_reference(multi):
    """Two almost collinear units (covariance eigenvalue ~1e-9, as for the first convolution of a
    WRN on CIFAR-100) make the precision ill-conditioned; float32 scores must still match a
    float64 computation."""
    g = torch.Generator().manual_seed(0)

    def sample(n):
        y = torch.arange(n) % 5
        z = torch.randn(n, 16, generator=g) + 3 * centers[y]
        z[:, -2] = z[:, -1] + 3e-5 * torch.randn(n, generator=g)
        return z, y

    centers = torch.randn(5, 16, generator=g)
    z, y = sample(500)
    x, _ = sample(64)
    if multi:
        scores = (
            MultiMahalanobis([torch.nn.Identity()]).fit_structured([z], y).predict_structured([x])
        )
    else:
        scores = Mahalanobis(None).fit_features(z, y).predict_features(x)
    expected = _reference_scores(z, y, x, eps=1e-10, use_pinv=False)
    # Scoring runs in the feature dtype: projecting float32 features of size ~3 that differ by ~3e-5
    # onto a whitening direction of weight ~3e4 loses about 1e-3 (relative) to cancellation. A float32
    # quadratic form with the precision matrix is off by a factor of ~200 here.
    torch.testing.assert_close(scores.double(), expected, rtol=1e-2, atol=1e-2)


def test_distances_of_wide_features_are_summed_accurately():
    """With an identity covariance only the summation over the feature dimensions can lose
    precision; float32 scores must be correctly rounded results of the float64 sums."""
    d = 1024
    centers = torch.stack([torch.full((d,), 3.0), torch.full((d,), -3.0)])
    z, y = points_with_covariance(centers, torch.ones(d))
    x = centers[0] + torch.randn(64, d, generator=torch.Generator().manual_seed(0))
    detector = Mahalanobis(None, eps=0.0).fit_features(z, y)
    expected = 0.5 * (x.double() - centers[0].double()).square().sum(dim=1)
    torch.testing.assert_close(detector.predict_features(x).double(), expected, rtol=2e-7, atol=0)


def test_feature_maps_are_averaged_spatially():
    """Feature maps of shape B x C x H x W score like their spatial means."""
    g = torch.Generator().manual_seed(0)
    y = torch.arange(60) % 3
    detector = Mahalanobis(None).fit_features(torch.randn(60, 8, generator=g), y)
    x = torch.randn(5, 8, 4, 3, generator=g)
    torch.testing.assert_close(
        detector.predict_features(x), detector.predict_features(x.mean((2, 3)))
    )


def test_pinv_discards_exactly_the_eigenvalues_below_its_cutoff():
    """``use_pinv`` drops the directions whose variance is below ``D * finfo(float64).eps`` times the
    largest variance, as :func:`torch.linalg.pinv` does: here 5e-16 is dropped (cutoff 8.9e-16) and
    1e-14 is kept."""
    variances = torch.tensor([1.0, 1e-8, 1e-14, 5e-16])
    z, y = points_with_covariance(torch.tensor([[0.0, 0, 0, 0], [2.0, 0, 0, 0]]), variances)
    detector = Mahalanobis(None, use_pinv=True).fit_features(z, y)
    expected = torch.diag(torch.tensor([1.0, 1e8, 1e14, 0.0]))
    torch.testing.assert_close(detector.precision, expected, rtol=1e-5, atol=0)
    pinv = torch.linalg.pinv(detector.cov.double(), hermitian=True).float()
    torch.testing.assert_close(detector.precision, pinv, rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    "features",
    [
        pytest.param(lambda z: z.index_fill(1, torch.tensor([0]), 0.0), id="dead-unit"),
        pytest.param(lambda z: torch.zeros_like(z), id="constant-features"),
    ],
)
def test_eps_zero_with_zero_variance_raises(features):
    """Without regularization, a unit without variance gives a covariance eigenvalue of exactly 0;
    this raises instead of producing NaN scores. (Covariances that are singular only up to rounding,
    e.g. with a duplicated unit, can have a tiny positive eigenvalue and do not raise.)"""
    g = torch.Generator().manual_seed(0)
    z = features(torch.randn(40, 6, generator=g))
    y = torch.arange(40) % 2
    with pytest.raises(ValueError, match="use_pinv"):
        Mahalanobis(None, eps=0.0).fit_features(z, y)


def test_parameters_set_by_hand_require_fit():
    """``mu`` and ``precision`` are results of fitting; setting them by hand does not fit."""
    detector = Mahalanobis(None)
    detector.mu = torch.zeros(2, 3)
    detector.precision = torch.eye(3)
    with pytest.raises(RequiresFittingException):
        detector.predict_features(torch.randn(4, 3))
