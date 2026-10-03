"""
test_sets of all benchmarks: ID only returns the ID test set once, OOD only one set per entry of
ood_names, and both one combined set per entry of ood_names. The datasets are small stand-ins,
so nothing is downloaded.
"""

import unittest

import torch
from torch.utils.data import TensorDataset

from src.pytorch_ood.benchmark import (
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
        bench._train_id, bench._test_id = TRAIN, TEST
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


class TestTestSets(unittest.TestCase):
    def test_id_only_returns_the_id_test_set_once(self):
        for cls in BENCHMARKS:
            with self.subTest(cls.__name__):
                sets = _benchmark(cls).test_sets(known=True, unknown=False)
                self.assertEqual(len(sets), 1)
                self.assertIs(sets[0], TEST)

    def test_ood_only_returns_one_set_per_ood_name(self):
        for cls in BENCHMARKS:
            with self.subTest(cls.__name__):
                sets = _benchmark(cls).test_sets(known=False, unknown=True)
                self.assertEqual([len(s) for s in sets], [3, 4])

    def test_both_combine_the_id_test_set_with_each_ood_set(self):
        for cls in BENCHMARKS:
            with self.subTest(cls.__name__):
                sets = _benchmark(cls).test_sets(known=True, unknown=True)
                self.assertEqual([len(s) for s in sets], [13, 14])

    def test_neither_raises(self):
        for cls in BENCHMARKS:
            with self.subTest(cls.__name__):
                with self.assertRaises(ValueError):
                    _benchmark(cls).test_sets(known=False, unknown=False)


if __name__ == "__main__":
    unittest.main()
