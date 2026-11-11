import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.detector import KNN


def _kth_distance(z_train, z_test, k, normalize=True):
    """Distance to the k-th nearest training feature, after L2 normalization (paper, Alg. 1)."""
    if normalize:
        z_train = z_train / z_train.norm(dim=1, keepdim=True)
        z_test = z_test / z_test.norm(dim=1, keepdim=True)
    return torch.cdist(z_test.double(), z_train.double()).sort(dim=1).values[:, k - 1]


def test_unknown_kwargs_raise():
    """There are no backend kwargs on the detector; backends are configured via ``search``."""
    with pytest.raises(TypeError):
        KNN(None, k=3, metric="euclidean")


@pytest.mark.parametrize("k", (1, 7, 50))
def test_matches_paper(k):
    g = torch.Generator().manual_seed(0)
    dtype = torch.float64
    z_train = torch.randn(300, 16, generator=g, dtype=dtype)
    z_train = z_train * torch.rand(300, 1, generator=g, dtype=dtype) * 10
    z_test = torch.randn(40, 16, generator=g, dtype=dtype) * 5
    labels = torch.zeros(300, dtype=torch.long)
    scores = KNN(None, k=k).fit_features(z_train, labels).predict_features(z_test)
    torch.testing.assert_close(scores, _kth_distance(z_train, z_test, k))


def test_without_normalization():
    g = torch.Generator().manual_seed(2)
    z_train = torch.randn(200, 8, generator=g, dtype=torch.float64) * 3
    z_test = torch.randn(30, 8, generator=g, dtype=torch.float64)
    labels = torch.zeros(200, dtype=torch.long)
    detector = KNN(None, k=5, normalize=False).fit_features(z_train, labels)
    torch.testing.assert_close(
        detector.predict_features(z_test), _kth_distance(z_train, z_test, 5, normalize=False)
    )


def test_zero_vectors_give_finite_scores():
    z_train = torch.randn(20, 4)
    z_train[3] = 0
    detector = KNN(None, k=3).fit_features(z_train, torch.zeros(20, dtype=torch.long))
    z_test = torch.cat([torch.zeros(1, 4), torch.randn(4, 4)])
    assert torch.isfinite(detector.predict_features(z_test)).all()


def test_fit_features_moves_unpinned_reference_to_detector_device():
    detector = KNN(torch.nn.Linear(4, 4).to("meta"), k=3)
    detector.fit_features(torch.randn(20, 4), torch.zeros(20, dtype=torch.long))
    assert detector.search.reference.device.type == "meta"


def test_sequential_encoder():
    """An encoder that evaluates to False, like an empty Sequential, is still an encoder."""
    detector = KNN(torch.nn.Sequential(), k=3)
    detector.fit(DataLoader(TensorDataset(torch.randn(20, 4), torch.zeros(20, dtype=torch.long))))
    assert detector.predict(torch.randn(5, 4)).shape == (5,)
