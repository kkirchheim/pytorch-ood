import re

import pytest
from torch.hub import HASH_REGEX

from pytorch_ood.model import (
    ImagePreprocessing,
    get_model_info,
    list_models,
    load_transform,
)
from pytorch_ood.model.registry import _ENTRIES


class TestRegistryIntegrity:
    """
    Offline consistency checks for all registry entries.
    """

    def test_keys_unique(self):
        keys = [e.key for e in _ENTRIES]
        assert len(keys) == len(set(keys))

    def test_slots_match_key(self):
        for entry in _ENTRIES:
            parts = [entry.arch, entry.dataset, entry.loss]
            if entry.seed is not None:
                parts.append(entry.seed)
            assert entry.key == "/".join(parts)

    def test_urls(self):
        for entry in _ENTRIES:
            assert entry.url.startswith("https://"), entry.key

    def test_sha256_format(self):
        for entry in _ENTRIES:
            assert re.search(r"^[a-f0-9]{64}$", entry.sha256), entry.key

    def test_file_name_hash_check(self):
        # torch.hub verifies the hash embedded in the cache filename
        for entry in _ENTRIES:
            match = HASH_REGEX.search(entry.file_name)
            assert match is not None, entry.key
            assert entry.sha256.startswith(match.group(1)), entry.key

    def test_arch_constructible(self):
        for entry in _ENTRIES:
            self._construct(entry)

    @staticmethod
    def _construct(entry):
        import importlib

        module_name, _, class_name = entry.arch_class.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**entry.arch_kwargs)


def test_exact_key():
    entry = get_model_info("wrn-40-2/cifar10/logitnorm/s0")
    assert entry.key == "wrn-40-2/cifar10/logitnorm/s0"


def test_exact_key_without_seed():
    entry = get_model_info("wrn-40-2/cifar10/oe")
    assert entry.key == "wrn-40-2/cifar10/oe"
    assert entry.seed is None


def test_prefix_resolution():
    entry = get_model_info("wrn-40-2/imagenet32")
    assert entry.key == "wrn-40-2/imagenet32/crossentropy"


@pytest.mark.parametrize(
    "prefix", ["wrn-40-2/cifar10", "wrn-40-2/cifar10/logitnorm", "wrn-40-2/cifar10/energy"]
)
def test_ambiguous_prefix_raises(prefix: str):
    with pytest.raises(ValueError):
        get_model_info(prefix)


def test_unknown():
    with pytest.raises(ValueError):
        get_model_info("resnet-18/mnist/crossentropy")


def test_list_models_all():
    keys = list_models()
    assert keys == sorted(keys)
    assert len(keys) == len(_ENTRIES)


def test_list_models_filtered():
    for key in list_models(dataset="cifar10"):
        assert get_model_info(key).dataset == "cifar10"

    assert list_models(dataset="no-such-dataset") == []


def test_build_structure():
    transform = load_transform("wrn-40-2/cifar10/crossentropy")
    names = [type(t).__name__ for t in transform.transforms]
    assert names == ["Resize", "ToRGB", "ToTensor", "Normalize"]


def test_std_introspection():
    # detectors like ODIN require access to the normalization std
    entry = get_model_info("wrn-40-2/cifar10/crossentropy")
    assert isinstance(entry.preprocessing, ImagePreprocessing)
    assert list(entry.preprocessing.std) == [x / 255 for x in [63.0, 62.1, 66.7]]


def test_all_entries_have_preprocessing():
    for entry in _ENTRIES:
        assert entry.preprocessing is not None, entry.key
