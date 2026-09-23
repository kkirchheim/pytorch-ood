import importlib.util
import inspect
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import ModelNotSetException, RequiresFittingException
from src.pytorch_ood.detector import NECO
from src.pytorch_ood.utils import OODMetrics
from tests.helpers import ClassificationModel


def anisotropic_train_features() -> torch.Tensor:
    """
    Six points in :math:`\\mathbb{R}^3` with exactly zero mean whose covariance is
    ``diag(2a^2, 2b^2, 2c^2) / 5`` with ``a > b > c``. The principal components are therefore
    exactly the canonical basis vectors, in the order ``e_0, e_1, e_2``.
    """
    a, b, c = 8.0, 4.0, 1.0
    return torch.tensor(
        [
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, -b, 0.0],
            [0.0, 0.0, c],
            [0.0, 0.0, -c],
        ]
    )


def correlated_train_features() -> torch.Tensor:
    """
    Four points with exactly zero mean, per-dimension variances 2.5 and 250 and a Pearson
    correlation of 0.6. After standardization the covariance is the correlation matrix
    ``[[1, 0.6], [0.6, 1]]``, whose leading eigenvector is ``[1, 1] / sqrt(2)`` with
    eigenvalue 1.6 (the trailing one is 0.4), so the standardized subspace is known exactly
    while the two raw dimensions differ in scale by a factor of ten.
    """
    return torch.tensor([[2.0, 20.0], [-2.0, -20.0], [1.0, -10.0], [-1.0, 10.0]])


class NegativeHead(torch.nn.Module):
    """
    Head whose maximum logit is the constant -1.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        return torch.cat([-2.0 * ones, -ones], dim=-1)


class ConstantHead(torch.nn.Module):
    """
    Head whose maximum logit is a known, per-sample quantity: ``sum(z) + offset``.
    """

    def __init__(self, dim: int = 3, offset: float = 10.0):
        super().__init__()
        self.dim = dim
        self.offset = offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        big = z.sum(dim=-1, keepdim=True) + self.offset
        small = big - 1.0
        return torch.cat([small, big], dim=-1)


class TestNECOScore(unittest.TestCase):
    """
    Exact, analytically computable tests of the NECO score.
    """

    def setUp(self) -> None:
        self.z_train = anisotropic_train_features()
        self.y_train = torch.zeros(self.z_train.shape[0], dtype=torch.long)

    def _fit(self, **kwargs) -> NECO:
        kwargs.setdefault("standardize", False)
        kwargs.setdefault("use_max_logit", False)
        detector = NECO(encoder=None, head=None, **kwargs)
        detector.fit_features(self.z_train, self.y_train)
        return detector

    def test_exact_score_first_component(self):
        """
        With d=1 the subspace is spanned by e_0, so the ratio for (3, 4, 0) is 3/5.
        A sign error, a wrong reduction axis, or a wrong subspace dimension all break this.
        """
        detector = self._fit(d=1)
        score = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0]]))
        self.assertAlmostEqual(score.item(), -0.6, places=5)

    def test_exact_score_two_components(self):
        """
        With d=2 the sample (3, 4, 0) lies entirely inside the subspace -> ratio 1.
        """
        detector = self._fit(d=2)
        score = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0]]))
        self.assertAlmostEqual(score.item(), -1.0, places=5)

    def test_component_order_is_descending(self):
        """
        The subspace must use the *largest* eigenvalues. If the order were ascending, the
        d=1 subspace would be e_2 and the score for (0, 0, 5) would be -1 instead of 0.
        """
        detector = self._fit(d=1)
        score = detector.predict_features(torch.tensor([[0.0, 0.0, 5.0]]))
        self.assertAlmostEqual(score.item(), 0.0, places=5)

    def test_outliers_receive_higher_scores(self):
        """
        Sign convention: samples orthogonal to the ETF subspace must score *higher*.
        """
        detector = self._fit(d=2)
        z = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        scores = detector.predict_features(z)
        self.assertAlmostEqual(scores[0].item(), -1.0, places=5)
        self.assertAlmostEqual(scores[1].item(), 0.0, places=5)
        self.assertGreater(scores[1].item(), scores[0].item())

    def test_features_are_centered_before_projection(self):
        """
        The official implementation projects with sklearn's PCA, which subtracts the training
        mean. Shifting the training data by mu must shift the projection, while the denominator
        stays the norm of the uncentered sample.
        """
        mu = torch.tensor([10.0, 0.0, 0.0])
        detector = NECO(encoder=None, head=None, d=1, standardize=False, use_max_logit=False)
        detector.fit_features(self.z_train + mu, self.y_train)

        x = torch.tensor([[13.0, 4.0, 0.0]])
        expected = -3.0 / torch.tensor([13.0, 4.0, 0.0]).norm().item()
        score = detector.predict_features(x)
        self.assertAlmostEqual(score.item(), expected, places=5)

        # without centering the numerator would be 13 instead of 3
        self.assertNotAlmostEqual(score.item(), -13.0 / 185**0.5, places=3)

    def test_max_logit_multiplication(self):
        """
        With ``use_max_logit=True`` the ratio is multiplied by the maximum logit.
        """
        head = ConstantHead()
        z = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])

        plain = self._fit(d=1)
        scaled = NECO(
            encoder=None, head=head, d=1, standardize=False, use_max_logit=True
        ).fit_features(self.z_train, self.y_train)

        max_logit = head(z).max(dim=-1).values
        expected = plain.predict_features(z) * max_logit

        torch.testing.assert_close(scaled.predict_features(z), expected)
        # the maximum logits here are 17 and 15, so the values really do differ
        self.assertFalse(torch.allclose(scaled.predict_features(z), plain.predict_features(z)))

    def test_max_logit_uses_raw_features(self):
        """
        Logits must be computed from the raw features, not from the standardized ones.
        """
        head = ConstantHead()
        z = torch.tensor([[3.0, 4.0, 0.0]])

        detector = NECO(encoder=None, head=head, d=1, standardize=True, use_max_logit=True)
        detector.fit_features(self.z_train, self.y_train)

        ratio = NECO(
            encoder=None, head=None, d=1, standardize=True, use_max_logit=False
        ).fit_features(self.z_train, self.y_train)

        expected = ratio.predict_features(z) * head(z).max(dim=-1).values
        torch.testing.assert_close(detector.predict_features(z), expected)

    def test_standardization_makes_score_scale_invariant(self):
        """
        Standardization removes per-dimension affine transformations of the feature space,
        so the score must be invariant when the same transformation is applied to train and
        test features. This fails if the scaler is not fitted on the training data, if it is
        not applied at prediction time, or if it is applied only to the numerator.
        """
        scale = torch.tensor([0.1, 7.0, 3.0])
        shift = torch.tensor([-2.0, 5.0, 1.0])
        z = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0], [1.0, -2.0, 3.0]])

        plain = NECO(encoder=None, head=None, d=1, standardize=True, use_max_logit=False)
        plain.fit_features(self.z_train, self.y_train)

        transformed = NECO(encoder=None, head=None, d=1, standardize=True, use_max_logit=False)
        transformed.fit_features(self.z_train * scale + shift, self.y_train)

        torch.testing.assert_close(
            plain.predict_features(z),
            transformed.predict_features(z * scale + shift),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_exact_score_standardized(self):
        """
        Hand-computed absolute value for the *default* standardized path, so that the strongest
        evidence for the default configuration does not depend on scikit-learn being installed.

        Standardizing divides the two dimensions by sqrt(2.5) and sqrt(250), mapping the probe
        (3, 10) to (1.8973666, 0.6324555). The leading component is [1, 1]/sqrt(2), so the ratio
        is (1.8973666 + 0.6324555) / sqrt(2) / 2.0 = 0.8944272.
        """
        detector = NECO(encoder=None, head=None, d=1, standardize=True, use_max_logit=False)
        detector.fit_features(correlated_train_features(), torch.zeros(4, dtype=torch.long))

        score = detector.predict_features(torch.tensor([[3.0, 10.0]]))
        self.assertAlmostEqual(score.item(), -0.8944272, places=5)

        # (3, -30) standardizes to (1.8973666, -1.8973666), which is orthogonal to the
        # leading component, so it projects to zero
        orthogonal = detector.predict_features(torch.tensor([[3.0, -30.0]]))
        self.assertAlmostEqual(orthogonal.item(), 0.0, places=5)

    def test_negative_max_logit_inverts_order(self):
        """
        Pins the documented consequence of the published formulation: a negative maximum logit
        flips the ordering. A well-meaning clamp of the logit would break this.
        """
        detector = NECO(
            encoder=None, head=NegativeHead(), d=2, standardize=False, use_max_logit=True
        )
        detector.fit_features(self.z_train, self.y_train)

        scores = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]))
        # score = -(ratio * -1) = ratio, so the inlier ends up with the *higher* score
        self.assertAlmostEqual(scores[0].item(), 1.0, places=5)
        self.assertAlmostEqual(scores[1].item(), 0.0, places=5)
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_constant_feature_dimension_is_handled(self):
        """
        A dead unit has zero variance; without the zero-variance guard standardization would
        divide by zero and every score would become NaN.
        """
        z_train = torch.cat([self.z_train, torch.full((6, 1), 2.0)], dim=1)
        detector = NECO(encoder=None, head=None, d=2, standardize=True, use_max_logit=False)
        detector.fit_features(z_train, self.y_train)

        scores = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0, 2.0]]))
        self.assertTrue(torch.isfinite(scores).all())

    def test_zero_norm_sample_is_finite(self):
        """
        A sample that standardizes to the zero vector must not produce 0/0.
        """
        detector = NECO(encoder=None, head=None, d=1, standardize=True, use_max_logit=False)
        detector.fit_features(self.z_train, self.y_train)

        score = detector.predict_features(detector.feature_mean.unsqueeze(0))
        self.assertTrue(torch.isfinite(score).all())
        self.assertAlmostEqual(score.item(), 0.0, places=5)

    def test_single_sample_fit_raises(self):
        detector = NECO(encoder=None, head=None, d=1, standardize=False, use_max_logit=False)
        with self.assertRaises(ValueError):
            detector.fit_features(self.z_train[:1], self.y_train[:1])

    def test_accepts_float64_input(self):
        """
        Features may arrive in double precision; the score is computed and returned in float32.
        """
        reference = self._fit(d=2)
        expected = reference.predict_features(torch.tensor([[3.0, 4.0, 0.0]]))

        detector = NECO(encoder=None, head=None, d=2, standardize=False, use_max_logit=False)
        detector.fit_features(self.z_train.double(), self.y_train)
        actual = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64))

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_standardize_flag_changes_result(self):
        """
        Guards against ``standardize`` being silently ignored.
        """
        z = torch.tensor([[3.0, 4.0, 0.0]])

        with_scaler = NECO(encoder=None, head=None, d=1, standardize=True, use_max_logit=False)
        with_scaler.fit_features(self.z_train, self.y_train)
        without = self._fit(d=1)

        self.assertFalse(
            torch.allclose(with_scaler.predict_features(z), without.predict_features(z))
        )

    def test_standardized_scores_are_bounded(self):
        """
        With standardization the denominator equals the norm of the full projection, hence
        the ratio lies in [0, 1] and the returned outlier score in [-1, 0].
        """
        torch.manual_seed(0)
        z_train = torch.randn(200, 8) * torch.arange(1, 9).float()
        detector = NECO(encoder=None, head=None, d=3, standardize=True, use_max_logit=False)
        detector.fit_features(z_train, torch.zeros(200, dtype=torch.long))

        scores = detector.predict_features(torch.randn(50, 8))
        self.assertTrue((scores <= 1e-5).all())
        self.assertTrue((scores >= -1 - 1e-5).all())

    def test_chunked_fit_matches_single_chunk(self):
        """
        The covariance is accumulated chunk-wise; the chunk size must not change the result.
        """
        torch.manual_seed(3)
        z_train = torch.randn(257, 6) * torch.arange(1, 7).float()
        y_train = torch.zeros(257, dtype=torch.long)
        z = torch.randn(9, 6)

        full = NECO(encoder=None, head=None, d=3, standardize=True, use_max_logit=False)
        full.fit_features(z_train, y_train, batch_size=1024)

        chunked = NECO(encoder=None, head=None, d=3, standardize=True, use_max_logit=False)
        chunked.fit_features(z_train, y_train, batch_size=16)

        torch.testing.assert_close(
            full.predict_features(z), chunked.predict_features(z), rtol=1e-4, atol=1e-6
        )

    def test_batch_independence(self):
        """
        Scores must not depend on the other samples in the batch.
        """
        torch.manual_seed(1)
        detector = self._fit(d=2)
        z = torch.randn(7, 3)

        batched = detector.predict_features(z)
        single = torch.cat([detector.predict_features(z[i : i + 1]) for i in range(z.shape[0])])
        torch.testing.assert_close(batched, single)

    def test_ood_samples_are_ignored_during_fit(self):
        """
        Samples with negative labels must not influence the estimated subspace.
        """
        z_ood = torch.tensor([[0.0, 0.0, 500.0], [0.0, 0.0, -500.0]])
        z = torch.cat([self.z_train, z_ood])
        y = torch.cat([self.y_train, -torch.ones(2, dtype=torch.long)])

        detector = NECO(encoder=None, head=None, d=1, standardize=False, use_max_logit=False)
        detector.fit_features(z, y)

        score = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0]]))
        self.assertAlmostEqual(score.item(), -0.6, places=5)

    def test_fit_without_id_samples_raises(self):
        detector = NECO(encoder=None, head=None, d=1, use_max_logit=False)
        with self.assertRaises(ValueError):
            detector.fit_features(self.z_train, -torch.ones(6, dtype=torch.long))

    def test_d_is_clamped_to_feature_dim(self):
        detector = self._fit(d=64)
        score = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0]]))
        self.assertAlmostEqual(score.item(), -1.0, places=5)

    def test_clamped_d_is_reported(self):
        """
        A clamped ``d`` must be visible, otherwise GridSearch records candidates that never
        described the computed projection.
        """
        detector = self._fit(d=64)
        self.assertEqual(detector.d, 3)
        self.assertEqual(detector.get_hyperparameters(), {"d": 3})
        self.assertIn("d=3", repr(detector))

        with self.assertLogs("src.pytorch_ood.detector.neco", level="WARNING"):
            detector.set_hyperparameters(d=100)
        self.assertEqual(detector.d, 3)

    def test_explained_variance_is_exposed(self):
        """
        The docstring tells users to pick ``d`` from the explained variance, so it must be
        available and correctly ordered. For this fixture the covariance is exactly
        ``diag(128, 32, 2) / 5``.
        """
        detector = self._fit(d=1)
        torch.testing.assert_close(
            detector.explained_variance,
            torch.tensor([25.6, 6.4, 0.4]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rank_deficient_fit_warns(self):
        """
        ``n`` centered samples span at most ``n - 1`` directions; asking for more must warn.
        """
        detector = NECO(encoder=None, head=None, d=5, standardize=False, use_max_logit=False)
        with self.assertLogs("src.pytorch_ood.detector.neco", level="WARNING") as ctx:
            detector.fit_features(torch.randn(3, 10), torch.zeros(3, dtype=torch.long))
        self.assertTrue(any("rank" in line for line in ctx.output))

    def test_d_can_be_changed_after_fit(self):
        """
        The full eigenbasis is retained, so GridSearch can tune ``d`` without refitting.
        """
        detector = self._fit(d=1)
        self.assertEqual(detector.get_hyperparameters(), {"d": 1})
        detector.set_hyperparameters(d=2)
        score = detector.predict_features(torch.tensor([[3.0, 4.0, 0.0]]))
        self.assertAlmostEqual(score.item(), -1.0, places=5)

    def test_invalid_d_raises(self):
        with self.assertRaises(ValueError):
            NECO(encoder=None, head=None, d=0, use_max_logit=False)


@unittest.skipUnless(
    importlib.util.find_spec("sklearn") is not None, "scikit-learn required for reference test"
)
class TestNECOReferenceEquivalence(unittest.TestCase):
    """
    Compares the detector against an independent NumPy/scikit-learn implementation of the
    scoring pipeline described by the paper and used by the official implementation
    (https://gitlab.com/drti/neco): standardize with statistics of the ID training features,
    project on the principal components of the standardized training features, take the norm
    of the first ``d`` coordinates relative to the norm of the standardized feature vector,
    and optionally multiply with the maximum logit.
    """

    def setUp(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        self.n_features, self.n_classes = 12, 5
        self.w = rng.normal(size=(self.n_classes, self.n_features))
        self.b = rng.normal(size=(self.n_classes,))
        scale = np.linspace(0.2, 5.0, self.n_features)
        self.z_train = rng.normal(size=(400, self.n_features)) * scale
        self.z_test = rng.normal(size=(37, self.n_features)) * scale + 0.3
        self.logits = self.z_test @ self.w.T + self.b

    def _head(self):
        import numpy as np

        w = torch.from_numpy(np.asarray(self.w)).float()
        b = torch.from_numpy(np.asarray(self.b)).float()
        return lambda z: z @ w.T + b

    def _reference(self, d: int, standardize: bool, use_max_logit: bool):
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        train, test = self.z_train, self.z_test
        if standardize:
            scaler = StandardScaler()
            train = scaler.fit_transform(train)
            test = scaler.transform(test)

        pca = PCA(n_components=self.n_features).fit(train)
        projected = pca.transform(test)[:, :d]

        score = np.linalg.norm(projected, axis=-1) / np.linalg.norm(test, axis=-1)
        if use_max_logit:
            score = score * self.logits.max(axis=-1)
        return score

    def _detector_scores(self, d: int, standardize: bool, use_max_logit: bool):
        detector = NECO(
            encoder=None,
            head=self._head() if use_max_logit else None,
            d=d,
            standardize=standardize,
            use_max_logit=use_max_logit,
        )
        detector.fit_features(
            torch.from_numpy(self.z_train).float(),
            torch.zeros(self.z_train.shape[0], dtype=torch.long),
        )
        # negated back into the "larger is more in-distribution" convention of the paper
        return -detector.predict_features(torch.from_numpy(self.z_test).float()).numpy()

    def test_matches_reference_with_standardization(self):
        import numpy as np

        for d in (1, 4, 8, 12):
            with self.subTest(d=d):
                expected = self._reference(d, standardize=True, use_max_logit=False)
                actual = self._detector_scores(d, standardize=True, use_max_logit=False)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_matches_reference_with_max_logit(self):
        import numpy as np

        expected = self._reference(4, standardize=True, use_max_logit=True)
        actual = self._detector_scores(4, standardize=True, use_max_logit=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_matches_reference_without_standardization(self):
        import numpy as np

        expected = self._reference(4, standardize=False, use_max_logit=False)
        actual = self._detector_scores(4, standardize=False, use_max_logit=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


class TestNECOAPI(unittest.TestCase):
    """
    Contract tests: fitting state, missing model/head, shapes, devices.
    """

    def setUp(self) -> None:
        torch.manual_seed(123)
        self.model = ClassificationModel(num_inputs=10, n_hidden=6, num_outputs=3).eval()
        x = torch.randn(40, 10)
        y = torch.arange(40) % 3
        self.loader = DataLoader(TensorDataset(x, y), batch_size=8)

    def test_requires_fit_flag(self):
        self.assertTrue(NECO.requires_fit)

    def test_documented_defaults(self):
        """
        The defaults are part of the documented contract: ``d=100`` and ``standardize=True``
        come from the reference implementation, and ``use_max_logit=True`` is the paper's
        headline configuration. Flipping any of them must be a deliberate diff.
        """
        params = inspect.signature(NECO.__init__).parameters
        self.assertEqual(params["d"].default, 100)
        self.assertIs(params["standardize"].default, True)
        self.assertIs(params["use_max_logit"].default, True)

    def test_hyperparameter_space_is_the_documented_one(self):
        self.assertEqual(set(NECO.hyperparameter_space), {"d"})
        candidates = NECO.hyperparameter_space["d"]
        self.assertIn(100, candidates, "the default must be reachable by GridSearch")
        self.assertEqual(candidates, sorted(candidates))
        # the paper's per-case optima span roughly 40 to 730 (Table C.5), so the grid has to
        # reach from well below the CIFAR-10/ViT optimum up into transformer-width territory
        self.assertLessEqual(min(candidates), 40)
        self.assertGreaterEqual(max(candidates), 512)

    def test_fit_ignores_ood_labelled_samples(self):
        """
        Repository convention: a training loader may contain ``ToUnknown()``-marked outliers,
        and they must not enter the estimated subspace.
        """
        torch.manual_seed(5)
        x_id = torch.randn(64, 10)
        y_id = torch.arange(64) % 3
        x_ood = torch.randn(64, 10) * 50.0 + 100.0
        y_ood = torch.full((64,), -1, dtype=torch.long)

        clean = DataLoader(TensorDataset(x_id, y_id), batch_size=16)
        contaminated = DataLoader(
            TensorDataset(torch.cat([x_id, x_ood]), torch.cat([y_id, y_ood])), batch_size=16
        )

        probe = torch.randn(8, 10)
        scores = []
        for loader in (clean, contaminated):
            detector = NECO(self.model.features, self.model.classifier, d=3)
            detector.fit(loader)
            scores.append(detector(probe))

        torch.testing.assert_close(scores[0], scores[1])

    def test_predict_without_fit_raises(self):
        detector = NECO(self.model.features, self.model.classifier, d=2)
        with self.assertRaises(RequiresFittingException):
            detector(torch.randn(4, 10))

    def test_predict_features_without_fit_raises(self):
        detector = NECO(None, None, d=2, use_max_logit=False)
        with self.assertRaises(RequiresFittingException):
            detector.predict_features(torch.randn(4, 6))

    def test_predict_without_encoder_raises(self):
        detector = NECO(None, None, d=2, use_max_logit=False)
        with self.assertRaises(ModelNotSetException):
            detector.predict(torch.randn(4, 10))

    def test_fit_without_encoder_raises(self):
        detector = NECO(None, None, d=2, use_max_logit=False)
        with self.assertRaises(ModelNotSetException):
            detector.fit(self.loader)

    def test_missing_head_raises(self):
        """
        A missing head is rejected at construction time, before an expensive fit, and the
        prediction-time guard still holds if the head is removed afterwards.
        """
        with self.assertRaises(ModelNotSetException):
            NECO(self.model.features, None, d=2, use_max_logit=True)

        detector = NECO(self.model.features, self.model.classifier, d=2, use_max_logit=True)
        detector.fit(self.loader)
        detector.head = None
        with self.assertRaises(ModelNotSetException):
            detector(torch.randn(4, 10))

    def test_non_grid_input_only(self):
        """
        NECO scores pooled feature vectors; grid-like feature maps are rejected explicitly.
        """
        detector = NECO(None, None, d=2, use_max_logit=False)
        with self.assertRaises(ValueError):
            detector.fit_features(torch.randn(8, 6, 2, 2), torch.zeros(8, dtype=torch.long))

        detector.fit_features(torch.randn(8, 6), torch.zeros(8, dtype=torch.long))
        with self.assertRaises(ValueError):
            detector.predict_features(torch.randn(4, 6, 2, 2))

    def test_end_to_end_shapes(self):
        detector = NECO(self.model.features, self.model.classifier, d=3)
        detector.fit(self.loader)

        scores = detector(torch.randn(11, 10))
        self.assertEqual(scores.shape, (11,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_predict_matches_predict_features(self):
        detector = NECO(self.model.features, self.model.classifier, d=3)
        detector.fit(self.loader)

        x = torch.randn(5, 10)
        with torch.no_grad():
            z = self.model.features(x)

        torch.testing.assert_close(detector(x), detector.predict_features(z))

    def test_no_gradients_are_tracked(self):
        detector = NECO(self.model.features, self.model.classifier, d=3)
        detector.fit(self.loader)
        scores = detector(torch.randn(5, 10, requires_grad=True))
        self.assertFalse(scores.requires_grad)

    def test_state_follows_to_device(self):
        detector = NECO(self.model.features, self.model.classifier, d=2)
        detector.fit(self.loader)

        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                detector.to(device)
                for attr in ("feature_mean", "feature_std", "pca_mean", "components"):
                    state = getattr(detector, attr)
                    self.assertIsNotNone(state, attr)
                    self.assertEqual(state.device.type, torch.device(device).type)

                scores = detector(torch.randn(4, 10))
                self.assertEqual(scores.device.type, torch.device(device).type)

    def test_detects_low_rank_violations(self):
        """
        End-to-end check of the method's premise on data constructed independently of the
        detector: in-distribution features are drawn from a rank-4 Gaussian in a 20-dimensional
        space, outliers are isotropic in the full space and therefore mostly live outside the
        in-distribution subspace. Nothing here refers to the fitted basis.
        """
        torch.manual_seed(11)
        n_dim, rank = 20, 4

        basis = torch.linalg.qr(torch.randn(n_dim, rank))[0]  # (20, 4), orthonormal columns

        def low_rank(n: int) -> torch.Tensor:
            return torch.randn(n, rank) @ basis.T

        z_train = low_rank(512)
        z_id = low_rank(256)
        # isotropic outliers, scaled to the same expected norm as the ID features so that the
        # separation cannot come from a norm difference alone
        z_ood = torch.randn(256, n_dim) * (rank / n_dim) ** 0.5

        loader = DataLoader(
            TensorDataset(z_train, torch.arange(512) % 3),
            batch_size=128,
        )

        detector = NECO(torch.nn.Identity(), None, d=rank, use_max_logit=False)
        detector.fit(loader)

        metrics = OODMetrics()
        metrics.update(detector(z_id), torch.zeros(256, dtype=torch.long))
        metrics.update(detector(z_ood), -torch.ones(256, dtype=torch.long))
        self.assertGreater(metrics.compute()["AUROC"], 0.99)


if __name__ == "__main__":
    unittest.main()
