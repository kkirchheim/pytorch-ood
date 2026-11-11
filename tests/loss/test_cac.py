import torch

from pytorch_ood.loss import CACLoss


def test_distance_calc():
    criterion = CACLoss(n_classes=10)
    z = torch.randn(size=(128, 10))
    assert criterion.distance(z).shape == (128, 10)


def test_score():
    criterion = CACLoss(n_classes=10)
    z = torch.randn(size=(128, 10))
    distances = criterion.centers(z)
    s = criterion.score(distances)
    assert s.shape == (128,)
