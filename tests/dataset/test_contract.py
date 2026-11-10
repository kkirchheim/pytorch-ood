"""
Properties every dataset has, checked for each dataset in :data:`DATASETS` without real data.
"""

import inspect
import socket
from pathlib import Path
from typing import Dict

import pytest
from PIL import Image

from pytorch_ood.dataset import img, txt
from pytorch_ood.dataset.img.base import ImageDatasetBase
from tests.helpers.components import DATASETS

#: placeholder in :data:`REQUIRED` for the ``root`` directory of the test
ROOT = object()

#: constructor arguments without a default, other than ``root``
REQUIRED: Dict[type, dict] = {
    img.CIFAR10C: {"subset": "all"},
    img.CIFAR100C: {"subset": "all"},
    img.FishyScapes: {"cs_root": ROOT},
    img.ImageNetC: {"subset": "blur"},
    img.MNISTC: {"subset": "all", "split": "test"},
    img.MVTechAD: {"split": "test"},
    img.SegmentMeIfYouCan: {"subset": "RoadAnomaly21"},
    img.StreetHazards: {"subset": "test"},
    txt.WikiText2: {"split": "test"},
    txt.WikiText103: {"split": "test"},
}


def _parameters(cls: type):
    return inspect.signature(cls.__init__).parameters


def _ids(cls: type) -> str:
    return cls.__name__


def _construct(cls: type, root: Path, **overrides):
    kwargs = {k: str(root) if v is ROOT else v for k, v in REQUIRED.get(cls, {}).items()}
    return cls(str(root), **{**kwargs, **overrides})


class NetworkAttempt(BaseException):
    """Not an ``Exception``, so ``pytest.raises(...)`` for the expected error cannot swallow it."""


@pytest.fixture
def no_network(monkeypatch):
    def connect(sock, address):
        raise NetworkAttempt(address)

    monkeypatch.setattr(socket.socket, "connect", connect)


ROOTED = [cls for cls in DATASETS if "root" in _parameters(cls)]


@pytest.mark.parametrize(
    "cls", [cls for cls in DATASETS if "download" in _parameters(cls)], ids=_ids
)
def test_download_defaults_to_false(cls: type):
    assert _parameters(cls)["download"].default is False


def test_every_dataset_with_a_root_can_be_constructed_here():
    missing = [
        cls.__name__
        for cls in ROOTED
        for name, p in list(_parameters(cls).items())[2:]
        if p.default is inspect.Parameter.empty and name not in REQUIRED.get(cls, {})
    ]
    assert missing == []


@pytest.mark.parametrize("cls", ROOTED, ids=_ids)
def test_missing_data_raises_without_downloading(cls: type, tmp_path: Path, no_network):
    # FileNotFoundError for some datasets, RuntimeError for most; no decision on one type yet
    with pytest.raises((RuntimeError, OSError)):
        _construct(cls, tmp_path)


CHECKS_SUBSET_LATE = pytest.mark.xfail(
    strict=True, reason="subset is checked after the data, so missing data raises RuntimeError"
)
INVALID_SUBSET_BROKEN = {
    img.CIFAR10C: CHECKS_SUBSET_LATE,
    img.CIFAR100C: CHECKS_SUBSET_LATE,
    img.OpenImagesO: pytest.mark.xfail(strict=True, reason="an invalid subset fails an assert"),
}


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(cls, id=cls.__name__, marks=INVALID_SUBSET_BROKEN.get(cls, ()))
        for cls in ROOTED
        if "subset" in _parameters(cls)
    ],
)
def test_invalid_subset_raises_value_error(cls: type, tmp_path: Path, no_network):
    with pytest.raises(ValueError):
        _construct(cls, tmp_path, subset="no such subset")


@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in ROOTED
        if issubclass(cls, ImageDatasetBase)
        and isinstance(cls.filename, str)
        and isinstance(cls.md5hash, str)
    ],
    ids=_ids,
)
def test_corrupted_archive_raises_without_downloading(cls: type, tmp_path: Path, no_network):
    (tmp_path / cls.filename).write_bytes(b"not the archive")

    with pytest.raises(RuntimeError, match="corrupted"):
        _construct(cls, tmp_path)


#: datasets whose images lie directly in ``base_folder``
FLAT_FOLDER = [
    img.LSUNCrop,
    img.LSUNResize,
    img.TinyImageNetCrop,
    img.TinyImageNetResize,
    img.iNaturalist,
]


@pytest.mark.parametrize("cls", FLAT_FOLDER, ids=_ids)
def test_flat_folder_dataset_on_a_fake_tree(cls: type, tmp_path: Path, monkeypatch):
    folder = tmp_path / cls.base_folder
    folder.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (4, 4), (i, 0, 0)).save(folder / f"{i}.png")
    monkeypatch.setattr(cls, "_check_integrity", lambda self: True)

    dataset = _construct(cls, tmp_path)

    assert len(dataset) == 3
    x, y = dataset[0]
    assert isinstance(x, Image.Image)
    assert y == -1
    with pytest.raises(IndexError):
        dataset[len(dataset)]
