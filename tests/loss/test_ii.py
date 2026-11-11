import torch

from pytorch_ood.loss import IILoss


def test_distance_and_predict_have_one_row_per_sample_and_one_column_per_class():
    criterion = IILoss(n_classes=10, n_embedding=8)
    z = torch.randn(size=(128, 8))

    assert criterion.distance(z).shape == (128, 10)
    assert criterion.predict(z).shape == (128, 10)
