"""Rules every benchmark suite follows."""

import inspect
import socket
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from pytorch_ood import benchmark as benchmarks

SUITES = [
    getattr(benchmarks, name)
    for name in benchmarks.__all__
    if inspect.isclass(getattr(benchmarks, name))
    and issubclass(getattr(benchmarks, name), benchmarks.Benchmark)
    and not inspect.isabstract(getattr(benchmarks, name))
]


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.__name__)
def test_downloads_are_opt_in(suite: type, tmp_path, monkeypatch):
    parameters = inspect.signature(suite).parameters
    assert parameters["download"].default is False

    attempts = []

    def attempt(*args, **kwargs):
        attempts.append(args)
        raise OSError("blocked")

    # the autouse socket guard raises too, which pytest.raises below would swallow
    monkeypatch.setattr(socket.socket, "connect", attempt)
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=attempt))
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))

    with pytest.raises(RuntimeError):
        suite(str(tmp_path / "root"), lambda x: x)

    assert attempts == []


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "suite",
    [benchmarks.CIFAR10_ODIN, benchmarks.CIFAR100_ODIN]
    + [benchmarks.CUB_SSB, benchmarks.StanfordCars_SSB, benchmarks.Aircraft_SSB],
    ids=lambda s: s.__name__,
)
def test_download_flag_reaches_every_dataset(suite: type, flag: bool, monkeypatch):
    seen = []

    def record(*args, **kwargs):
        if "download" in kwargs:
            seen.append(kwargs["download"])
        return mock.MagicMock()

    def record_splits(dataset, root, download=False):
        seen.append(download)
        return mock.MagicMock()

    module = sys.modules[suite.__module__]
    for name, value in list(vars(module).items()):
        if inspect.isclass(value) and issubclass(value, Dataset):
            monkeypatch.setattr(module, name, record)
    if hasattr(module, "load_ssb_splits"):
        monkeypatch.setattr(module, "load_ssb_splits", record_splits)

    suite("root", lambda x: x, download=flag)

    assert seen and set(seen) == {flag}


@pytest.mark.parametrize(
    "suite", [benchmarks.CIFAR10_ODIN, benchmarks.CIFAR100_ODIN], ids=lambda s: s.__name__
)
def test_odin_noise_sets_are_seeded_and_named_alike(suite: type, monkeypatch):
    module = sys.modules[suite.__module__]
    for name, value in list(vars(module).items()):
        if inspect.isclass(value) and issubclass(value, Dataset) and "Noise" not in name:
            monkeypatch.setattr(module, name, mock.MagicMock())

    first, second = suite("root", lambda x: x), suite("root", lambda x: x)

    assert first.ood_names[-2:] == ["UniformNoise", "GaussianNoise"]
    for a, b in zip(first.test_oods[-2:], second.test_oods[-2:]):
        assert np.array_equal(np.asarray(a[0][0]), np.asarray(b[0][0]))
