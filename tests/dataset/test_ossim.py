import torch
from torch.utils.data import TensorDataset

from pytorch_ood.dataset.ossim import DynamicOSS


def test_something():
    x = torch.randn(size=(100, 10))
    y = torch.arange(100) % 10

    dataset = TensorDataset(x, y)

    ossim = DynamicOSS(dataset, kuc=0, uuc_test=4, uuc_val=0, seed=0)

    assert 6 == len(ossim.kkc)
    assert 4 == len(ossim.uuc)

    assert ossim.test_dataset() is not None
    assert ossim.train_dataset() is not None
    assert ossim.val_dataset() is not None
