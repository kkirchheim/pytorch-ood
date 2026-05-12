import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from pytorch_ood.benchmark.img.ssb import (
    _CUB200,
    _FGVCAircraft,
    _StanfordCars,
)
from pytorch_ood.utils import oscr_score


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_image(path, size=(32, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.zeros((*size, 3), dtype=np.uint8)).save(path)


def _make_cub_root(root, n_classes=10, n_per_class=4):
    """Create a minimal fake CUB-200-2011 directory structure."""
    base = os.path.join(root, "CUB_200_2011")
    os.makedirs(base, exist_ok=True)

    img_id = 1
    images_lines, labels_lines, split_lines = [], [], []

    for cls in range(1, n_classes + 1):
        cls_dir = f"{cls:03d}.Class_{cls}"
        for i in range(n_per_class):
            rel = f"{cls_dir}/img_{i}.jpg"
            _write_image(os.path.join(base, "images", rel))
            images_lines.append(f"{img_id} {rel}\n")
            labels_lines.append(f"{img_id} {cls}\n")
            # first half train, second half test
            split_lines.append(f"{img_id} {1 if i < n_per_class // 2 else 0}\n")
            img_id += 1

    with open(os.path.join(base, "images.txt"), "w") as f:
        f.writelines(images_lines)
    with open(os.path.join(base, "image_class_labels.txt"), "w") as f:
        f.writelines(labels_lines)
    with open(os.path.join(base, "train_test_split.txt"), "w") as f:
        f.writelines(split_lines)

    return root


def _make_aircraft_root(root, n_classes=10, n_per_class=4):
    """Create a minimal fake FGVC-Aircraft directory structure."""
    base = os.path.join(root, "fgvc-aircraft-2013b", "data")
    img_dir = os.path.join(base, "images")
    os.makedirs(img_dir, exist_ok=True)

    variants = [f"variant_{i}" for i in range(n_classes)]
    with open(os.path.join(base, "variants.txt"), "w") as f:
        f.write("\n".join(variants) + "\n")

    for split in ("train", "val", "trainval", "test"):
        lines = []
        for cls_idx, variant in enumerate(variants):
            for i in range(n_per_class):
                img_id = f"{cls_idx:04d}{i:02d}"
                _write_image(os.path.join(img_dir, f"{img_id}.jpg"))
                lines.append(f"{img_id} {variant}\n")
        with open(os.path.join(base, f"images_variant_{split}.txt"), "w") as f:
            f.writelines(lines)

    return root


def _make_cars_root(root, n_classes=10, n_per_class=4):
    """Create a minimal fake Stanford Cars directory structure using .mat files."""
    import scipy.io

    base = os.path.join(root, "stanford_cars")
    devkit = os.path.join(base, "devkit")
    os.makedirs(devkit, exist_ok=True)
    os.makedirs(os.path.join(base, "cars_train"), exist_ok=True)
    os.makedirs(os.path.join(base, "cars_test"), exist_ok=True)

    def _build_annos(img_dir, split):
        entries = []
        for cls_idx in range(1, n_classes + 1):
            for i in range(n_per_class):
                fname = f"{cls_idx:03d}_{i:02d}.jpg"
                _write_image(os.path.join(img_dir, fname))
                entries.append((0, 0, 32, 32, cls_idx, fname))
        # Build structured array matching Stanford Cars .mat format
        dt = np.dtype(
            [
                ("bbox_x1", "O"),
                ("bbox_y1", "O"),
                ("bbox_x2", "O"),
                ("bbox_y2", "O"),
                ("class", "O"),
                ("fname", "O"),
            ]
        )
        arr = np.empty(len(entries), dtype=dt)
        for i, (x1, y1, x2, y2, cls_idx, fname) in enumerate(entries):
            arr[i] = (
                np.array([[x1]]),
                np.array([[y1]]),
                np.array([[x2]]),
                np.array([[y2]]),
                np.array([[cls_idx]]),
                np.array([[fname]]),
            )
        return arr

    train_annos = _build_annos(os.path.join(base, "cars_train"), "train")
    test_annos = _build_annos(os.path.join(base, "cars_test"), "test")

    scipy.io.savemat(os.path.join(devkit, "cars_train_annos.mat"), {"annotations": train_annos})
    scipy.io.savemat(
        os.path.join(devkit, "cars_test_annos_withlabels.mat"), {"annotations": test_annos}
    )

    return root


# ─── _CUB200 Tests ────────────────────────────────────────────────────────────


class CUB200DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_cub_root(self.tmp, n_classes=10, n_per_class=4)

    def test_loads_all_classes(self):
        ds = _CUB200(self.tmp, split="train")
        # 10 classes × 2 train images each = 20
        self.assertEqual(len(ds), 20)

    def test_class_filter_reduces_size(self):
        ds = _CUB200(self.tmp, split="train", classes=[0, 1, 2])
        self.assertEqual(len(ds), 6)

    def test_class_filter_remaps_labels(self):
        ds = _CUB200(self.tmp, split="train", classes=[2, 5, 8])
        labels = {label for _, label in ds.data}
        self.assertEqual(labels, {0, 1, 2})

    def test_test_split_loaded(self):
        ds = _CUB200(self.tmp, split="test")
        self.assertEqual(len(ds), 20)

    def test_getitem_returns_image_and_label(self):
        ds = _CUB200(self.tmp, split="train", classes=[0])
        img, label = ds[0]
        self.assertIsInstance(label, int)

    def test_unknown_transform_gives_neg_one(self):
        from pytorch_ood.utils import ToUnknown

        ds = _CUB200(self.tmp, split="train", classes=[0], target_transform=ToUnknown())
        _, label = ds[0]
        self.assertEqual(label, -1)


# ─── _FGVCAircraft Tests ──────────────────────────────────────────────────────


class FGVCAircraftDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_aircraft_root(self.tmp, n_classes=10, n_per_class=4)

    def test_loads_all_classes(self):
        ds = _FGVCAircraft(self.tmp, split="test")
        self.assertEqual(len(ds), 40)

    def test_class_filter_reduces_size(self):
        ds = _FGVCAircraft(self.tmp, split="test", classes=[0, 1])
        self.assertEqual(len(ds), 8)

    def test_class_filter_remaps_labels(self):
        ds = _FGVCAircraft(self.tmp, split="test", classes=[3, 7])
        labels = {label for _, label in ds.data}
        self.assertEqual(labels, {0, 1})


# ─── _StanfordCars Tests ──────────────────────────────────────────────────────


class StanfordCarsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_cars_root(self.tmp, n_classes=10, n_per_class=4)

    def test_loads_all_classes(self):
        ds = _StanfordCars(self.tmp, split="train")
        self.assertEqual(len(ds), 40)

    def test_class_filter_reduces_size(self):
        ds = _StanfordCars(self.tmp, split="train", classes=[0, 1, 2])
        self.assertEqual(len(ds), 12)

    def test_class_filter_remaps_labels(self):
        ds = _StanfordCars(self.tmp, split="train", classes=[0, 4, 9])
        labels = {label for _, label in ds.data}
        self.assertEqual(labels, {0, 1, 2})


# ─── SSB Benchmark Structure Tests ───────────────────────────────────────────


class _MockSSBBenchmark:
    """Helper that builds a CUB_SSB-like structure from fake data."""

    def __init__(self, tmp):
        _make_cub_root(tmp, n_classes=10, n_per_class=4)

        # Mimic split structure: 6 known, 2 easy OOD, 2 hard OOD classes
        from pytorch_ood.utils import ToUnknown
        from pytorch_ood.benchmark.img.ssb import _SSBBase, _CUB200

        class FakeCUB_SSB(_SSBBase):
            def __init__(self):
                known = list(range(6))
                easy = [6, 7]
                hard = [8, 9]
                self._train = _CUB200(tmp, split="train", classes=known)
                self._test_id = _CUB200(tmp, split="test", classes=known)
                self._test_easy = _CUB200(
                    tmp, split="test", classes=easy, target_transform=ToUnknown()
                )
                self._test_hard = _CUB200(
                    tmp, split="test", classes=hard, target_transform=ToUnknown()
                )
                self.ood_names = ["Easy", "Hard"]

        self.bench = FakeCUB_SSB()


class SSBBenchmarkStructureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mock = _MockSSBBenchmark(self.tmp)
        self.bench = self.mock.bench

    def test_test_sets_returns_two_datasets(self):
        sets = self.bench.test_sets()
        self.assertEqual(len(sets), 2)

    def test_ood_names_are_easy_and_hard(self):
        self.assertEqual(self.bench.ood_names, ["Easy", "Hard"])

    def test_ood_labels_are_neg_one(self):
        # Easy set: concat of ID test + easy OOD
        ds = self.bench.test_sets()[0]
        labels = [label for _, label in ds]
        ood_labels = [label for label in labels if label < 0]
        self.assertTrue(all(label == -1 for label in ood_labels))
        self.assertGreater(len(ood_labels), 0)

    def test_id_labels_are_non_negative(self):
        ds = self.bench.test_sets()[0]
        id_labels = [label for _, label in ds if label >= 0]
        self.assertTrue(all(label >= 0 for label in id_labels))

    def test_train_set_has_no_ood_labels(self):
        ds = self.bench.train_set()
        labels = [label for _, label in ds]
        self.assertTrue(all(label >= 0 for label in labels))

    def test_known_only_returns_two_id_datasets(self):
        sets = self.bench.test_sets(known=True, unknown=False)
        self.assertEqual(len(sets), 2)
        for ds in sets:
            labels = [label for _, label in ds]
            self.assertTrue(all(label >= 0 for label in labels))

    def test_unknown_only_returns_two_ood_datasets(self):
        sets = self.bench.test_sets(known=False, unknown=True)
        self.assertEqual(len(sets), 2)
        for ds in sets:
            labels = [label for _, label in ds]
            self.assertTrue(all(label == -1 for label in labels))


# ─── OSCR Tests ───────────────────────────────────────────────────────────────


class OSCRTest(unittest.TestCase):
    def _make_data(self, n_id=100, n_ood=50, accuracy=1.0, seed=42):
        torch.manual_seed(seed)
        labels = torch.cat([torch.arange(n_id), torch.full((n_ood,), -1)])
        predictions = torch.cat([torch.arange(n_id), torch.zeros(n_ood, dtype=torch.long)])
        if accuracy < 1.0:
            n_wrong = int(n_id * (1 - accuracy))
            predictions[:n_wrong] = (predictions[:n_wrong] + 1) % n_id
        return labels, predictions

    def test_returns_float(self):
        labels, preds = self._make_data()
        scores = torch.randn(len(labels))
        result = oscr_score(scores, preds, labels)
        self.assertIsInstance(result, float)

    def test_range_zero_to_one(self):
        labels, preds = self._make_data()
        scores = torch.randn(len(labels))
        result = oscr_score(scores, preds, labels)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_perfect_detector_and_classifier(self):
        n_id, n_ood = 100, 50
        labels = torch.cat([torch.arange(n_id), torch.full((n_ood,), -1)])
        preds = torch.cat([torch.arange(n_id), torch.zeros(n_ood, dtype=torch.long)])
        # Perfect: ID scores all low, OOD scores all high
        scores = torch.cat([torch.zeros(n_id), torch.ones(n_ood)])
        result = oscr_score(scores, preds, labels)
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_worst_case_inverted_scores(self):
        n_id, n_ood = 100, 50
        labels = torch.cat([torch.arange(n_id), torch.full((n_ood,), -1)])
        preds = torch.cat([torch.arange(n_id), torch.zeros(n_ood, dtype=torch.long)])
        # Inverted: ID scores all high, OOD scores all low.
        # The curve jumps from (0,0) to (FPR=1, CCR≈0) in one step — OSCR ≈ 0.
        scores = torch.cat([torch.ones(n_id), torch.zeros(n_ood)])
        result = oscr_score(scores, preds, labels)
        self.assertLess(result, 1.0 / n_id)

    def test_raises_without_ood_samples(self):
        labels = torch.arange(10)
        scores = torch.randn(10)
        preds = torch.arange(10)
        with self.assertRaises(ValueError):
            oscr_score(scores, preds, labels)

    def test_raises_without_id_samples(self):
        labels = torch.full((10,), -1)
        scores = torch.randn(10)
        preds = torch.zeros(10, dtype=torch.long)
        with self.assertRaises(ValueError):
            oscr_score(scores, preds, labels)


if __name__ == "__main__":
    unittest.main()
