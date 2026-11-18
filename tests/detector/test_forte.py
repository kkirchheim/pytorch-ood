import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import Forte
from pytorch_ood.detector.forte import _knn_radii, _prdc

sklearn = pytest.importorskip("sklearn")

K = 5


def _features(n, d, seed, shift=0.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=torch.float64) + shift).to(dtype)


def _labels(n):
    return torch.zeros(n, dtype=torch.long)


def _radius(z, i, k):
    """Distance of point i to its k-th nearest neighbor among the other points."""
    others = torch.cat([z[:i], z[i + 1 :]])
    return (others - z[i]).norm(dim=1).sort().values[k - 1]


def _statistics(reference, z, k=K):
    """Recall, density, precision and coverage of every point, one point at a time."""
    n = len(reference)
    reference_radii = torch.stack([_radius(reference, i, k) for i in range(n)])
    rows = []
    for j in range(len(z)):
        distances = (reference - z[j]).norm(dim=1)
        radius = _radius(z, j, k)
        in_reference_balls = (distances < reference_radii).sum().item()
        rows.append(
            [
                (distances < radius).sum().item() / n,
                in_reference_balls / (k * n),
                float(in_reference_balls > 0),
                float(distances.min() < radius),
            ]
        )
    return torch.tensor(rows, dtype=torch.float64)


def _paper_scores(train, test, method="gmm", seed=0):
    """Scores as in the reference implementation, for a list of feature spaces."""
    from scipy.stats import gaussian_kde
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import train_test_split
    from sklearn.svm import OneClassSVM

    fit, evaluated = [], []
    for z_train, z_test in zip(train, test):
        reference, second = train_test_split(z_train.numpy(), test_size=0.5, random_state=seed)
        reference = torch.from_numpy(reference)
        fit.append(_statistics(reference, torch.from_numpy(second)))
        evaluated.append(_statistics(reference, z_test))
    fit, evaluated = torch.cat(fit, dim=1).numpy(), torch.cat(evaluated, dim=1).numpy()

    if method == "gmm":
        models = [
            GaussianMixture(n_components=n, random_state=seed).fit(fit)
            for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            if n < len(fit)
        ]
        best = models[int(np.argmax([m.score(fit) for m in models]))]
        return -best.score_samples(evaluated)
    if method == "kde":
        models = [gaussian_kde(fit.T, bw_method=bw) for bw in ("scott", "silverman")]
        best = models[int(np.argmax([m.logpdf(fit.T).mean() for m in models]))]
        return -best.logpdf(evaluated.T)
    models = [
        OneClassSVM(kernel="rbf", gamma="auto", nu=nu).fit(fit)
        for nu in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    ]
    best = models[int(np.argmax([(m.predict(fit) == 1).mean() for m in models]))]
    return -best.decision_function(evaluated)


@pytest.mark.parametrize("shift", (0.0, 1.0))
def test_statistics_match_definition(shift):
    reference = _features(60, 6, seed=0)
    z = _features(40, 6, seed=1, shift=shift)

    statistics = _prdc(reference, _knn_radii(reference, K), z, K)

    torch.testing.assert_close(statistics, _statistics(reference, z), rtol=0, atol=1e-12)


def test_statistics_of_float32_features_with_large_norms():
    # the neighborhoods are far smaller than the norms, which float32 distances do not resolve
    reference = _features(80, 32, seed=0, shift=300.0, dtype=torch.float32)
    z = _features(50, 32, seed=1, shift=300.0, dtype=torch.float32)
    labels = _labels(len(reference))

    single = Forte(None).fit_features(reference, labels)
    double = Forte(None).fit_features(reference.double(), labels)

    assert single._reference[0].dtype == torch.float64
    np.testing.assert_array_equal(single._statistics([z]), double._statistics([z.double()]))


@pytest.mark.parametrize("method", ("gmm", "kde", "ocsvm"))
def test_matches_paper(method):
    train = _features(120, 8, seed=0)
    test = torch.cat([_features(30, 8, seed=1), _features(30, 8, seed=2, shift=1.0)])

    detector = Forte(None, method=method).fit_features(train, _labels(len(train)))

    np.testing.assert_allclose(
        detector.predict_features(test).numpy(),
        _paper_scores([train], [test], method=method),
        rtol=1e-6,
        atol=1e-6,
    )


def test_matches_paper_with_several_encoders():
    train = [_features(120, 8, seed=0), _features(120, 5, seed=1)]
    test = [_features(50, 8, seed=2, shift=0.5), _features(50, 5, seed=3, shift=0.5)]

    detector = Forte(None, seed=3).fit_features(train, _labels(120))

    assert detector._statistics(test).shape == (50, 8)
    np.testing.assert_allclose(
        detector.predict_features(test).numpy(),
        _paper_scores(train, test, seed=3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_outliers_score_higher():
    train = _features(1000, 2, seed=0)
    test = torch.cat([_features(100, 2, seed=1), _features(100, 2, seed=2, shift=4.0)])

    scores = Forte(None).fit_features(train, _labels(len(train))).predict_features(test)

    assert scores[100:].min() > scores[:100].median()


def test_list_of_encoders():
    g = torch.Generator().manual_seed(0)
    x_train, x_test = torch.randn(64, 10, generator=g), torch.randn(16, 10, generator=g)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        encoders = [torch.nn.Linear(10, 6), torch.nn.Linear(10, 4)]
    loader = DataLoader(TensorDataset(x_train, _labels(64)), batch_size=16)

    detector = Forte(encoders).fit(loader)
    with torch.no_grad():
        twin = Forte(None).fit_features([e(x_train) for e in encoders], _labels(64))
        expected = twin.predict_features([e(x_test) for e in encoders])

    scores = detector(x_test)
    assert scores.shape == (16,)
    torch.testing.assert_close(scores, expected)


def test_number_of_feature_tensors_must_match_fit():
    detector = Forte(None).fit_features(_features(40, 4, seed=0), _labels(40))

    with pytest.raises(ValueError, match="1 encoder"):
        detector.predict_features([_features(8, 4, seed=1), _features(8, 4, seed=2)])


def test_feature_tensors_must_have_the_same_number_of_rows():
    with pytest.raises(ValueError, match="same inputs"):
        Forte(None).fit_features([_features(40, 4, seed=0), _features(30, 4, seed=1)], _labels(40))


def test_inputs_of_a_call_are_scored_as_one_sample():
    assert Forte.batch_dependent
    detector = Forte(None).fit_features(_features(200, 8, seed=0), _labels(200))
    z = torch.cat([_features(20, 8, seed=1), _features(20, 8, seed=2, shift=2.0)])

    scores = detector.predict_features(z)

    # the neighborhoods of the first half change without the second half
    assert not torch.allclose(detector.predict_features(z[:20]), scores[:20])
    # the order within the sample does not matter
    perm = torch.arange(len(z)).roll(1)
    torch.testing.assert_close(detector.predict_features(z[perm]), scores[perm])


def test_sample_with_k_points():
    reference = _features(40, 4, seed=0)
    z = _features(K, 4, seed=1)

    statistics = _prdc(reference, _knn_radii(reference, K), z, K)

    # the neighborhood of a point reaches to the farthest of the other points
    expected = _statistics(reference, z, k=K - 1)
    torch.testing.assert_close(statistics[:, [0, 3]], expected[:, [0, 3]], rtol=0, atol=1e-12)


def test_single_input_has_no_recall_and_coverage():
    reference = _features(40, 4, seed=0)

    statistics = _prdc(reference, _knn_radii(reference, K), reference[:1], K)

    assert statistics[0, [0, 3]].tolist() == [0, 0]


def test_split_depends_on_seed():
    train = _features(100, 4, seed=0)
    first, again, other = (
        Forte(None, seed=seed).fit_features(train, _labels(100)) for seed in (0, 0, 1)
    )

    assert torch.equal(first._reference[0], again._reference[0])
    assert not torch.equal(first._reference[0], other._reference[0])
    assert len(first._reference[0]) == 50


def test_too_few_samples():
    with pytest.raises(ValueError, match="at least 4"):
        Forte(None).fit_features(_features(3, 4, seed=0), _labels(3))


def test_unknown_method():
    with pytest.raises(ValueError, match="method"):
        Forte(None, method="flow")
