"""
test_sets of all benchmarks: ID only returns the ID test set once, OOD only one set per entry of
ood_names, and both one combined set per entry of ood_names. The datasets are small stand-ins,
so nothing is downloaded.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from pytorch_ood.benchmark import (
    CIFAR10_ODIN,
    CIFAR100_ODIN,
    CUB_SSB,
    CIFAR10_OpenOOD,
    CIFAR100_OpenOOD,
    MIDOG_OpenMIBOOD,
)


def _dataset(n, label):
    return TensorDataset(torch.zeros(n, 1), torch.full((n,), label))


TRAIN, TEST = _dataset(50, 0), _dataset(10, 0)
OODS = [_dataset(3, -1), _dataset(4, -1)]


def _benchmark(cls):
    """A benchmark object whose datasets are the stand-ins above."""
    bench = object.__new__(cls)
    if issubclass(cls, CUB_SSB):
        bench._train, bench._test_id = TRAIN, TEST
        bench._test_easy, bench._test_hard = OODS
    else:
        bench.train_in, bench.test_in, bench.test_oods = TRAIN, TEST, OODS
    bench.ood_names = ["a", "b"]
    return bench


BENCHMARKS = (
    CIFAR10_ODIN,
    CIFAR100_ODIN,
    CIFAR10_OpenOOD,
    CIFAR100_OpenOOD,
    MIDOG_OpenMIBOOD,
    CUB_SSB,
)


@pytest.mark.parametrize("cls", BENCHMARKS)
def test_train_set_is_the_id_training_set(cls):
    assert _benchmark(cls).train_set() is TRAIN


@pytest.mark.parametrize("cls", BENCHMARKS)
def test_id_only_returns_the_id_test_set_once(cls):
    sets = _benchmark(cls).test_sets(known=True, unknown=False)
    assert len(sets) == 1
    assert sets[0] is TEST


@pytest.mark.parametrize("cls", BENCHMARKS)
def test_ood_only_returns_one_set_per_ood_name(cls):
    sets = _benchmark(cls).test_sets(known=False, unknown=True)
    assert [len(s) for s in sets] == [3, 4]


@pytest.mark.parametrize("cls", BENCHMARKS)
def test_both_combine_the_id_test_set_with_each_ood_set(cls):
    sets = _benchmark(cls).test_sets(known=True, unknown=True)
    assert [len(s) for s in sets] == [13, 14]


@pytest.mark.parametrize("cls", BENCHMARKS)
def test_neither_raises(cls):
    with pytest.raises(ValueError):
        _benchmark(cls).test_sets(known=False, unknown=False)
