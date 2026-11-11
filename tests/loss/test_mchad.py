import torch

from pytorch_ood.loss import MCHADLoss


def test_distance_has_one_row_per_sample_and_one_column_per_class():
    criterion = MCHADLoss(n_classes=10, n_dim=5)

    assert criterion.distance(torch.randn(size=(128, 5))).shape == (128, 10)
