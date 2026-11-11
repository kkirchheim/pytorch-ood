import pytest
import torch

from pytorch_ood.loss import BackgroundClassLoss


def test_class_to_high():
    criterion = BackgroundClassLoss(n_classes=3)
    logits = torch.randn(size=(10, 3))
    target = torch.arange(10) % 3
    target[5:] = 10

    with pytest.raises(ValueError):
        criterion(logits, target)
