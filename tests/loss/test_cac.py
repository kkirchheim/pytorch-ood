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


def test_loss_matches_reference():
    # tuplet term sums over all classes j != y, including the last class
    torch.manual_seed(0)
    n_classes, alpha = 5, 0.7
    criterion = CACLoss(n_classes=n_classes, alpha=alpha)
    distances = torch.rand(12, n_classes) * 3
    target = torch.tensor([0, 1, 2, 3, 4, 4, 0, 2, 4, 1, 3, 4])

    expected = torch.zeros(())
    for d, y in zip(distances, target):
        others = [j for j in range(n_classes) if j != y]
        expected += alpha * d[y] + torch.log(1 + sum(torch.exp(d[y] - d[j]) for j in others))
    expected /= len(target)

    assert torch.allclose(criterion(distances, target), expected, atol=1e-5)
