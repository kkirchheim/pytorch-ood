"""
Noise datasets: each sample is a function of the seed and the index only, so it does not depend on
access order, worker process or epoch.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from pytorch_ood.dataset.img import GaussianNoise, UniformNoise

NOISE = (GaussianNoise, UniformNoise)


def _noise(cls, seed=0, length=8):
    return cls(length, size=(4, 4, 3), seed=seed)


def _array(sample) -> np.ndarray:
    return np.asarray(sample[0])


@pytest.mark.parametrize("cls", NOISE)
def test_sample_depends_only_on_seed_and_index(cls):
    ds = _noise(cls)
    first = _array(ds[3])
    for i in (0, 1, 2):
        ds[i]
    assert np.array_equal(_array(ds[3]), first)
    assert np.array_equal(_array(ds[-5]), first)
    assert not np.array_equal(_array(ds[2]), first)

    assert np.array_equal(_array(_noise(cls, seed=0)[3]), first)
    assert not np.array_equal(_array(_noise(cls, seed=1)[3]), first)

    with pytest.raises(IndexError):
        ds[len(ds)]


@pytest.mark.parametrize("cls", NOISE)
def test_seed_none_is_fixed_at_construction(cls):
    ds = _noise(cls, seed=None)
    assert np.array_equal(_array(ds[0]), _array(ds[0]))


@pytest.mark.parametrize("cls", NOISE)
def test_workers_give_the_samples_drawn_without_workers(cls):
    ds = _noise(cls)
    ds.transform = ToTensor()
    expected = torch.stack([ds[i][0] for i in range(len(ds))])

    loader = DataLoader(ds, batch_size=2, num_workers=2)
    loaded = torch.cat([x for x, _ in loader])

    assert torch.equal(loaded, expected)
    assert len(torch.unique(loaded.flatten(1), dim=0)) == len(ds)
