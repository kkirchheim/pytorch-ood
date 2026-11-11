"""
Tests for the OpenMIBOOD benchmarks.

These tests use synthetic 8x8 PNG fixtures and exercise the structural contract of
the benchmark classes (split counts, ood_names ordering, train/test loading, custom
loader for non-PIL formats). They do not download any real medical imaging data.
"""

import contextlib
import os
import tempfile
import unittest
from os.path import join
from unittest import mock

from PIL import Image
from torchvision import transforms

from pytorch_ood.benchmark import (
    MIDOG_OpenMIBOOD,
    OASIS3_OpenMIBOOD,
    PhaKIR_OpenMIBOOD,
)
from pytorch_ood.dataset.img import ImageListDataset


def _write_png(path, color=(128, 64, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _write_imglist(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for relpath, label in entries:
            f.write(f"{relpath} {label}\n")


class _FixtureBase:
    """
    Builds a self-contained data root + an in-process resource directory whose
    layout matches what the benchmark classes expect, then patches
    ``openmibood_module._RESOURCES`` to point at it.
    """

    benchmark_cls = None
    subdir = None
    num_classes = None
    expected_train_size = 4
    expected_test_in_size = 2
    expected_ood_per_split = 1

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_root = cls._tmp.name
        cls.data_root = join(cls.tmp_root, "data")
        os.makedirs(cls.data_root, exist_ok=True)

        # Build train + test ID + every OOD split
        cls._build_fixtures()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _build_fixtures(cls):
        bench = cls.benchmark_cls
        bench_resource_dir = join(cls.data_root, "imglists", cls.subdir)

        # Train set: enough samples to cover all classes
        train_entries = [
            (f"train_{i}.png", i % cls.num_classes) for i in range(cls.expected_train_size)
        ]
        for rel, _ in train_entries:
            _write_png(join(cls.data_root, rel))
        _write_imglist(join(bench_resource_dir, bench._train_imglist), train_entries)

        # Validation ID set
        val_entries = [(f"val_{i}.png", i % cls.num_classes) for i in range(3)]
        for rel, _ in val_entries:
            _write_png(join(cls.data_root, rel))
        _write_imglist(join(bench_resource_dir, bench._val_imglist), val_entries)

        # Test ID set
        test_entries = [
            (f"test_{i}.png", i % cls.num_classes) for i in range(cls.expected_test_in_size)
        ]
        for rel, _ in test_entries:
            _write_png(join(cls.data_root, rel))
        _write_imglist(join(bench_resource_dir, bench._test_imglist), test_entries)

        # One sample per OOD imglist
        for i, fname in enumerate(bench._ood_imglists):
            relpath = f"ood_{i}.png"
            _write_png(join(cls.data_root, relpath))
            _write_imglist(
                join(bench_resource_dir, fname),
                [(relpath, 0)],  # label is overwritten by ToUnknown anyway
            )


class _BenchmarkContractMixin:
    """Shared assertions for all 3 benchmarks."""

    def _make_benchmark(self, **kwargs):
        return self.benchmark_cls(
            root=self.data_root,
            transform=transforms.ToTensor(),
            **kwargs,
        )

    def test_categorical_attributes(self):
        bench = self._make_benchmark()
        self.assertEqual(
            bench.ood_names,
            list(bench.cs_id_names) + list(bench.near_ood_names) + list(bench.far_ood_names),
        )
        self.assertEqual(len(bench.test_oods), len(bench.ood_names))

    def test_train_set_returns_id_train(self):
        bench = self._make_benchmark()
        train = bench.train_set()
        self.assertEqual(len(train), self.expected_train_size)

    def test_ood_samples_get_unknown_label(self):
        bench = self._make_benchmark()
        ood = bench.test_oods[0]
        _, label = ood[0]
        self.assertEqual(label, -1)

    def test_id_samples_keep_real_label(self):
        bench = self._make_benchmark()
        _, label = bench.test_in[0]
        self.assertGreaterEqual(label, 0)

    def test_default_does_not_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("pytorch_ood.benchmark.img.openmibood.download_url") as download_url:
                with self.assertRaises(RuntimeError):
                    self.benchmark_cls(root=tmp, transform=transforms.ToTensor())
                download_url.assert_not_called()

    def test_val_set(self):
        bench = self._make_benchmark()
        self.assertEqual(len(bench.val_set()), 3)
        _, label = bench.val_set()[0]
        self.assertGreaterEqual(label, 0)

    @contextlib.contextmanager
    def _without(self, relpath):
        """
        Temporarily removes a fixture file.
        """
        path = join(self.data_root, relpath)
        os.rename(path, path + ".bak")
        try:
            yield
        finally:
            os.rename(path + ".bak", path)

    def test_missing_ood_data_raises(self):
        with self._without("ood_0.png"):
            with self.assertRaises(FileNotFoundError) as ctx:
                self._make_benchmark()
        all_names = self.benchmark_cls.cs_id_names + self.benchmark_cls.near_ood_names
        self.assertIn(all_names[0], str(ctx.exception))
        self.assertIn("skip_missing", str(ctx.exception))

    def test_skip_missing_ood_data(self):
        with self._without("ood_0.png"):
            with self.assertLogs("pytorch_ood.benchmark.img.openmibood", "WARNING") as logs:
                bench = self._make_benchmark(skip_missing=True)
        first = self.benchmark_cls.cs_id_names[0]
        self.assertIn(first, logs.output[0])
        self.assertNotIn(first, bench.ood_names)
        self.assertEqual(len(bench.ood_names), len(bench.test_oods))
        self.assertEqual(len(bench.test_sets()), len(bench._ood_imglists) - 1)

    def test_missing_id_data_raises(self):
        with self._without("train_0.png"):
            with self.assertRaises(FileNotFoundError):
                self._make_benchmark(skip_missing=True)

    def test_download_only_fetches_missing_downloadable_sources(self):
        def make(name, download, present):
            source = mock.Mock()
            source.name = name
            source.download = download
            source.is_present.return_value = present
            return source

        missing = make("missing", mock.Mock(), present=False)
        present = make("present", mock.Mock(), present=True)
        manual = make("manual", None, present=False)
        with mock.patch.object(self.benchmark_cls, "_sources", [missing, present, manual]):
            self._make_benchmark(download=True)
        missing.download.assert_called_once_with(self.data_root)
        present.download.assert_not_called()

    def test_download_errors_propagate(self):
        source = mock.Mock()
        source.download.side_effect = RuntimeError("MD5 mismatch")
        source.is_present.return_value = False
        with mock.patch.object(self.benchmark_cls, "_sources", [source]):
            with self.assertRaises(RuntimeError):
                self._make_benchmark(download=True)


class MIDOGOpenMIBOODTest(_FixtureBase, _BenchmarkContractMixin, unittest.TestCase):
    benchmark_cls = MIDOG_OpenMIBOOD
    subdir = "midog"
    num_classes = 3

    def test_split_counts(self):
        bench = self._make_benchmark()
        self.assertEqual(len(bench.cs_id_names), 2)
        self.assertEqual(len(bench.near_ood_names), 7)
        self.assertEqual(len(bench.far_ood_names), 2)
        self.assertEqual(len(bench.ood_names), 11)


class PhaKIROpenMIBOODTest(_FixtureBase, _BenchmarkContractMixin, unittest.TestCase):
    benchmark_cls = PhaKIR_OpenMIBOOD
    subdir = "phakir"
    num_classes = 7

    def test_split_counts(self):
        bench = self._make_benchmark()
        self.assertEqual(len(bench.cs_id_names), 2)
        self.assertEqual(len(bench.near_ood_names), 3)
        self.assertEqual(len(bench.far_ood_names), 2)
        self.assertEqual(len(bench.ood_names), 7)


class OASIS3OpenMIBOODTest(_FixtureBase, _BenchmarkContractMixin, unittest.TestCase):
    benchmark_cls = OASIS3_OpenMIBOOD
    subdir = "oasis3"
    num_classes = 2

    def test_split_counts(self):
        bench = self._make_benchmark()
        self.assertEqual(len(bench.cs_id_names), 2)
        self.assertEqual(len(bench.near_ood_names), 3)
        self.assertEqual(len(bench.far_ood_names), 2)
        self.assertEqual(len(bench.ood_names), 7)

    def test_custom_loader_is_used(self):
        # Verify that the loader callable is wired through to the underlying datasets
        sentinel = {"called": 0}

        def fake_loader(path):
            sentinel["called"] += 1
            return Image.new("RGB", (8, 8))

        bench = self._make_benchmark(loader=fake_loader)
        _ = bench.test_in[0]
        self.assertGreater(sentinel["called"], 0)


class ImageListDatasetTest(unittest.TestCase):
    def test_loads_and_skips_blank_and_comment_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_png(join(tmp, "a.png"))
            _write_png(join(tmp, "b.png"))
            imglist = join(tmp, "list.txt")
            with open(imglist, "w") as f:
                f.write("# header comment\n")
                f.write("\n")
                f.write("a.png 0\n")
                f.write("b.png 1\n")

            ds = ImageListDataset(tmp, imglist)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.labels, [0, 1])

    def test_missing_root_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            imglist = join(tmp, "list.txt")
            with open(imglist, "w") as f:
                f.write("a.png 0\n")
            with self.assertRaises(RuntimeError):
                ImageListDataset(join(tmp, "does-not-exist"), imglist)

    def test_missing_imglist_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ImageListDataset(tmp, join(tmp, "missing.txt"))


if __name__ == "__main__":
    unittest.main()
