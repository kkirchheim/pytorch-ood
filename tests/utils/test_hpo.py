import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import Detector, FeaturesDetector
from pytorch_ood.detector import ASH, GEN, KNN, ODIN, ReAct
from pytorch_ood.metrics import AUROC, FPRAtTPR, MetricCollection
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import GridSearch
from tests.helpers import ClassificationModel


def _make_val_dataset(seed=123):
    """
    Two-column features: column 0 perfectly separates ID (low) from OOD (high),
    column 1 is pure noise. A detector that weights column 0 is optimal.
    """
    g = torch.Generator().manual_seed(seed)
    n = 200
    col0_id = torch.randn(n, 1, generator=g) * 0.1
    col0_ood = torch.randn(n, 1, generator=g) * 0.1 + 5.0
    noise_id = torch.randn(n, 1, generator=g)
    noise_ood = torch.randn(n, 1, generator=g)

    x_id = torch.cat([col0_id, noise_id], dim=1)
    x_ood = torch.cat([col0_ood, noise_ood], dim=1)
    y_id = torch.zeros(n, dtype=torch.long)  # ID
    y_ood = torch.full((n,), -1, dtype=torch.long)  # OOD

    x = torch.cat([x_id, x_ood])
    y = torch.cat([y_id, y_ood])
    return TensorDataset(x, y)


def _make_fit_dataset(seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.cat(
        [torch.randn(100, 1, generator=g) * 0.1, torch.randn(100, 1, generator=g)], dim=1
    )
    y = torch.zeros(100, dtype=torch.long)
    return TensorDataset(x, y)


class _WeightedFeatureDetector(FeaturesDetector):
    """
    Synthetic feature detector. Score is a convex combination of the separating
    column and the noise column, controlled by hyperparameter ``w``. AUROC is
    maximized at ``w = 0`` (use only the separating column).
    """

    requires_fit = True
    hyperparameter_space = {"w": [0.0, 0.25, 0.5, 0.75, 1.0]}

    def __init__(self):
        self.encoder = lambda x: x  # features are the raw inputs
        self.w = 0.5
        self.fit_calls = 0

    def fit_features(self, x, y):
        self.fit_calls += 1
        return self

    def predict_features(self, x):
        return (1.0 - self.w) * x[:, 0] + self.w * x[:, 1]

    def predict(self, x):
        return self.predict_features(self.encoder(x))


class _RawWeightedDetector(Detector):
    """Same logic but with no cacheable producer, forcing the raw fallback path."""

    hyperparameter_space = {"w": [0.0, 0.25, 0.5, 0.75, 1.0]}

    def __init__(self):
        self.w = 0.5

    def predict(self, x):
        return (1.0 - self.w) * x[:, 0] + self.w * x[:, 1]


class _FitAffectingDetector(FeaturesDetector):
    """Hyperparameter ``bias`` affects the fitted state (like ReAct's percentile)."""

    requires_fit = True
    hyperparameter_space = {"bias": [0.0, 1.0, 2.0]}

    def __init__(self):
        self.encoder = lambda x: x
        self.bias = 0.0
        self._fitted_bias = None

    def fit_features(self, x, y):
        self._fitted_bias = self.bias  # derived state captured at fit time
        return self

    def predict_features(self, x):
        return x[:, 0] + self._fitted_bias  # a constant shift => AUROC ties

    def predict(self, x):
        return self.predict_features(self.encoder(x))


class _NaNDetector(FeaturesDetector):
    """Returns NaN for ``w == 0`` (the first candidate) and a separable score otherwise."""

    requires_fit = False
    hyperparameter_space = {"w": [0.0, 1.0]}

    def __init__(self, always_nan=False):
        self.encoder = lambda x: x
        self.w = 0.0
        self.always_nan = always_nan

    def predict_features(self, x):
        if self.always_nan or self.w == 0.0:
            return torch.full((x.shape[0],), float("nan"))
        return x[:, 0]  # separable column

    def predict(self, x):
        return self.predict_features(self.encoder(x))


class _ConstantDetector(Detector):
    """Returns constant scores for ``w = 1``, for which AUTC is undefined."""

    hyperparameter_space = {"w": [0.0, 1.0]}

    def __init__(self):
        self.w = 0.0

    def predict(self, x):
        return (1.0 - self.w) * x[:, 0]


class TestGridSearch:
    def setup_method(self):
        self.val_loader = DataLoader(_make_val_dataset(), batch_size=64)
        self.fit_loader = DataLoader(_make_fit_dataset(), batch_size=64)

    def test_finds_optimum_cached(self):
        detector = _WeightedFeatureDetector()
        search = GridSearch(detector, self.fit_loader, self.val_loader)
        best = search.run()

        assert best == {"w": 0.0}
        assert search.best_score_ > 0.95
        # detector left configured with the best params
        assert detector.w == 0.0

    def test_refits_per_candidate(self):
        detector = _WeightedFeatureDetector()
        search = GridSearch(detector, self.fit_loader, self.val_loader)
        search.run()
        # one fit per candidate, plus one final re-fit to leave the detector
        # consistently fitted for the best params
        assert detector.fit_calls == len(detector.hyperparameter_space["w"]) + 1

    def test_direction_flag(self):
        # minimizing AUROC should pick the worst (noise-only) weight
        detector = _WeightedFeatureDetector()
        search = GridSearch(detector, self.fit_loader, self.val_loader, higher_is_better=False)
        best = search.run()
        assert best == {"w": 1.0}

    def test_raw_fallback(self):
        detector = _RawWeightedDetector()
        search = GridSearch(detector, None, self.val_loader)
        best = search.run()
        assert best == {"w": 0.0}
        assert search.best_score_ > 0.95

    def test_records_all_results(self):
        detector = _WeightedFeatureDetector()
        search = GridSearch(detector, self.fit_loader, self.val_loader)
        search.run()
        assert len(search.results_) == len(detector.hyperparameter_space["w"])

    def test_explicit_space_override(self):
        detector = _WeightedFeatureDetector()
        search = GridSearch(
            detector, self.fit_loader, self.val_loader, hyperparameter_space={"w": [0.0, 1.0]}
        )
        search.run()
        assert len(search.results_) == 2

    def test_empty_space_raises(self):
        detector = _RawWeightedDetector()
        detector.hyperparameter_space = {}
        with pytest.raises(ValueError):
            GridSearch(detector, None, self.val_loader)


def test_get_set_roundtrip():
    detector = _WeightedFeatureDetector()
    detector.set_hyperparameters(w=0.25)
    assert detector.get_hyperparameters() == {"w": 0.25}


def test_unknown_hyperparameter_raises():
    detector = _WeightedFeatureDetector()
    with pytest.raises(ValueError):
        detector.set_hyperparameters(does_not_exist=1)


def test_ash_exposes_space():
    model = WideResNet(num_classes=10).eval()
    detector = ASH(backbone=model.feature_maps, head=model.forward_feature_maps)
    assert "percentile" in detector.hyperparameter_space
    detector.set_hyperparameters(percentile=0.9)
    assert detector.percentile == 0.9
    assert detector.get_hyperparameters() == {"percentile": 0.9}


def test_gen_exposes_space():
    detector = GEN(ClassificationModel())
    assert set(detector.hyperparameter_space) == {"gamma", "M"}
    detector.set_hyperparameters(gamma=2, M=100)
    assert detector.get_hyperparameters() == {"gamma": 2, "M": 100}


def test_odin_exposes_space():
    detector = ODIN(ClassificationModel())
    assert set(detector.hyperparameter_space) == {"temperature", "eps"}
    detector.set_hyperparameters(temperature=100, eps=0.0028)
    assert detector.get_hyperparameters() == {"temperature": 100, "eps": 0.0028}


class TestGridSearchReAct:
    """ReAct is a fitting feature-map detector: tuning re-fits via the raw path."""

    def _val_dataset(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        z_id = torch.randn(100, 10, generator=g)
        z_ood = torch.randn(100, 10, generator=g) + 4.0
        x = torch.cat([z_id, z_ood])
        y = torch.cat([torch.zeros(100, dtype=torch.long), torch.full((100,), -1)])
        return TensorDataset(x, y)

    def test_tunes_percentile(self):
        model = ClassificationModel(num_inputs=10).eval()
        detector = ReAct(backbone=model.features, head=model.classifier)

        fit_loader = DataLoader(
            TensorDataset(torch.randn(100, 10), torch.zeros(100, dtype=torch.long)),
            batch_size=32,
        )
        val_loader = DataLoader(self._val_dataset(), batch_size=64)

        search = GridSearch(detector, fit_loader, val_loader)
        best = search.run()

        assert "percentile" in best
        assert best["percentile"] in ReAct.hyperparameter_space["percentile"]
        assert len(search.results_) == len(ReAct.hyperparameter_space["percentile"])
        # detector left fitted with a threshold derived from the winning percentile
        assert detector.threshold is not None


def test_tunes_k():
    g = torch.Generator().manual_seed(0)
    z_id_train = torch.randn(80, 10, generator=g)
    fit_loader = DataLoader(
        TensorDataset(z_id_train, torch.zeros(80, dtype=torch.long)), batch_size=32
    )
    z_id = torch.randn(40, 10, generator=g)
    z_ood = torch.randn(40, 10, generator=g) + 4.0
    x = torch.cat([z_id, z_ood])
    y = torch.cat([torch.zeros(40, dtype=torch.long), torch.full((40,), -1)])
    val_loader = DataLoader(TensorDataset(x, y), batch_size=64)

    detector = KNN(encoder=lambda z: z)
    # small space so n_neighbors stays below the number of fitted samples
    search = GridSearch(detector, fit_loader, val_loader, hyperparameter_space={"k": [1, 3, 5]})
    best = search.run()

    assert best["k"] in [1, 3, 5]
    assert len(search.results_) == 3
    assert detector.k == best["k"]


def test_tunes_gamma_and_M():
    model = ClassificationModel(num_inputs=10, num_outputs=10).eval()
    detector = GEN(model)

    g = torch.Generator().manual_seed(0)
    z_id = torch.randn(100, 10, generator=g)
    z_ood = torch.randn(100, 10, generator=g) + 4.0
    x = torch.cat([z_id, z_ood])
    y = torch.cat([torch.zeros(100, dtype=torch.long), torch.full((100,), -1)])
    val_loader = DataLoader(TensorDataset(x, y), batch_size=64)

    # no fit_loader needed: GEN does not require fitting
    search = GridSearch(detector, None, val_loader)
    best = search.run()

    space = GEN.hyperparameter_space
    assert set(best) == {"gamma", "M"}
    assert best["gamma"] in space["gamma"]
    assert best["M"] in space["M"]
    # full Cartesian product of the two sweeps
    assert len(search.results_) == len(space["gamma"]) * len(space["M"])
    assert (detector.gamma, detector.M) == (best["gamma"], best["M"])


class TestGridSearchEdgeCases:
    def setup_method(self):
        self.val_loader = DataLoader(_make_val_dataset(), batch_size=64)
        self.fit_loader = DataLoader(_make_fit_dataset(), batch_size=64)

    def test_empty_candidate_list_raises(self):
        with pytest.raises(ValueError):
            GridSearch(
                _RawWeightedDetector(), None, self.val_loader, hyperparameter_space={"w": []}
            )

    def test_override_space_for_empty_class_space(self):
        # a detector with no declared space is tunable purely via the override argument
        detector = _RawWeightedDetector()
        detector.hyperparameter_space = {}
        search = GridSearch(
            detector, None, self.val_loader, hyperparameter_space={"w": [0.0, 1.0]}
        )
        best = search.run()
        assert best == {"w": 0.0}

    def test_requires_fit_without_fit_loader_raises(self):
        # _WeightedFeatureDetector.requires_fit is True
        search = GridSearch(_WeightedFeatureDetector(), None, self.val_loader)
        with pytest.raises(ValueError):
            search.run()

    def test_non_finite_score_is_not_selected(self):
        # first candidate (w=0) scores NaN; the search must still pick the finite one
        detector = _NaNDetector()
        search = GridSearch(detector, None, self.val_loader)
        best = search.run()
        assert best == {"w": 1.0}
        assert math.isfinite(search.best_score_)

    def test_all_non_finite_raises(self):
        detector = _NaNDetector(always_nan=True)
        search = GridSearch(detector, None, self.val_loader)
        with pytest.raises(ValueError):
            search.run()

    def test_detector_left_consistently_fitted(self):
        # fitted state must match the *best* params, not the last candidate evaluated
        detector = _FitAffectingDetector()
        search = GridSearch(detector, self.fit_loader, self.val_loader)
        best = search.run()
        assert detector.bias == best["bias"]
        assert detector._fitted_bias == detector.bias

    def test_metric_collection(self):
        metric = MetricCollection([AUROC(), FPRAtTPR()])
        for detector, fit_loader in (
            (_WeightedFeatureDetector(), self.fit_loader),
            (_RawWeightedDetector(), None),
        ):
            search = GridSearch(detector, fit_loader, self.val_loader, metric=metric)
            assert search.run() == {"w": 0.0}

    def test_constant_scores_do_not_abort(self):
        search = GridSearch(_ConstantDetector(), None, self.val_loader)
        with pytest.warns(UserWarning):
            best = search.run()
        assert best == {"w": 0.0}
        assert search.results_[1]["score"] == 0.5

    def test_unknown_metric_name_raises_readable(self):
        detector = _RawWeightedDetector()
        search = GridSearch(detector, None, self.val_loader, metric_name="NOPE")
        with pytest.raises(ValueError) as ctx:
            search.run()
        assert "NOPE" in str(ctx.value)
