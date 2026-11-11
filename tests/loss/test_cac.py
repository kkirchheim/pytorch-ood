import torch

from pytorch_ood.loss import CACLoss


class TestCAC:
    def test_distance_calc(self):
        criterion = CACLoss(n_classes=10)
        z = torch.randn(size=(128, 10))
        assert criterion.distance(z).shape == (128, 10)

    def test_score(self):
        criterion = CACLoss(n_classes=10)
        z = torch.randn(size=(128, 10))
        distances = criterion.centers(z)
        s = criterion.score(distances)
        assert s.shape == (128,)
