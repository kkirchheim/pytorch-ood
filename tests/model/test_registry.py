import unittest

from torch.hub import HASH_REGEX

from src.pytorch_ood.model import (
    ImagePreprocessing,
    get_model_info,
    list_models,
    load_transform,
)
from src.pytorch_ood.model.registry import _ENTRIES


class TestRegistryIntegrity(unittest.TestCase):
    """
    Offline consistency checks for all registry entries.
    """

    def test_keys_unique(self):
        keys = [e.key for e in _ENTRIES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_slots_match_key(self):
        for entry in _ENTRIES:
            parts = [entry.arch, entry.dataset, entry.loss]
            if entry.seed is not None:
                parts.append(entry.seed)
            self.assertEqual(entry.key, "/".join(parts))

    def test_urls(self):
        for entry in _ENTRIES:
            self.assertTrue(entry.url.startswith("https://"), entry.key)

    def test_sha256_format(self):
        for entry in _ENTRIES:
            self.assertRegex(entry.sha256, r"^[a-f0-9]{64}$", entry.key)

    def test_file_name_hash_check(self):
        # torch.hub verifies the hash embedded in the cache filename
        for entry in _ENTRIES:
            match = HASH_REGEX.search(entry.file_name)
            self.assertIsNotNone(match, entry.key)
            self.assertTrue(entry.sha256.startswith(match.group(1)), entry.key)

    def test_arch_constructible(self):
        for entry in _ENTRIES:
            model = self._construct(entry)
            self.assertIsNotNone(model)

    @staticmethod
    def _construct(entry):
        import importlib

        module_name, _, class_name = entry.arch_class.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**entry.arch_kwargs)


class TestRegistryLookup(unittest.TestCase):
    def test_exact_key(self):
        entry = get_model_info("wrn-40-2/cifar10/logitnorm/s0")
        self.assertEqual(entry.key, "wrn-40-2/cifar10/logitnorm/s0")

    def test_exact_key_without_seed(self):
        entry = get_model_info("wrn-40-2/cifar10/oe")
        self.assertEqual(entry.key, "wrn-40-2/cifar10/oe")
        self.assertIsNone(entry.seed)

    def test_prefix_resolution(self):
        entry = get_model_info("wrn-40-2/cifar10/logitnorm")
        self.assertEqual(entry.key, "wrn-40-2/cifar10/logitnorm/s0")

    def test_ambiguous(self):
        with self.assertRaises(ValueError):
            get_model_info("wrn-40-2/cifar10")

    def test_unknown(self):
        with self.assertRaises(ValueError):
            get_model_info("resnet-18/mnist/crossentropy")

    def test_list_models_all(self):
        keys = list_models()
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(_ENTRIES))

    def test_list_models_filtered(self):
        for key in list_models(dataset="cifar10"):
            self.assertEqual(get_model_info(key).dataset, "cifar10")

        self.assertEqual(list_models(dataset="no-such-dataset"), [])


class TestPreprocessing(unittest.TestCase):
    def test_build_structure(self):
        transform = load_transform("wrn-40-2/cifar10/crossentropy")
        names = [type(t).__name__ for t in transform.transforms]
        self.assertEqual(names, ["Resize", "ToRGB", "ToTensor", "Normalize"])

    def test_std_introspection(self):
        # detectors like ODIN require access to the normalization std
        entry = get_model_info("wrn-40-2/cifar10/crossentropy")
        self.assertIsInstance(entry.preprocessing, ImagePreprocessing)
        self.assertEqual(list(entry.preprocessing.std), [x / 255 for x in [63.0, 62.1, 66.7]])

    def test_all_entries_have_preprocessing(self):
        for entry in _ENTRIES:
            self.assertIsNotNone(entry.preprocessing, entry.key)


if __name__ == "__main__":
    unittest.main()
