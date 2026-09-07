import unittest
from unittest import mock

from pytorch_ood.dataset.img import ImageNet200
from pytorch_ood.dataset.img.base import _get_resource_file
from pytorch_ood.dataset.img.imagenet200 import _load_classes


def _read_split_resource(name):
    """Return list of (basename, label) from a split resource file."""
    entries = []
    with open(_get_resource_file(name)) as f:
        for line in f:
            line = line.strip()
            if line:
                base, label = line.rsplit(" ", 1)
                entries.append((base, int(label)))
    return entries


class _FakeImageNet:
    """Stand-in for torchvision.datasets.ImageNet exposing .samples and .wnids."""

    def __init__(self, samples, wnids):
        self.samples = samples
        self.wnids = wnids


def _fake_imagenet_factory(train=None, val=None):
    def factory(root, split):
        return train if split == "train" else val

    return factory


CLASSES = _load_classes()


class ImageNet200ResourceTest(unittest.TestCase):
    def test_classes_count_and_order(self):
        self.assertEqual(len(CLASSES), 200)
        self.assertEqual(len(set(CLASSES)), 200)
        # OpenOOD assigns labels in sorted WNID order
        self.assertEqual(CLASSES, sorted(CLASSES))

    def test_test_split_size_and_balance(self):
        entries = _read_split_resource("imagenet200_test.txt")
        self.assertEqual(len(entries), 9000)
        labels = [lbl for _, lbl in entries]
        self.assertEqual(set(labels), set(range(200)))
        self.assertTrue(all(labels.count(i) == 45 for i in range(200)))

    def test_val_split_size_and_balance(self):
        entries = _read_split_resource("imagenet200_val.txt")
        self.assertEqual(len(entries), 1000)
        labels = [lbl for _, lbl in entries]
        self.assertEqual(set(labels), set(range(200)))
        self.assertTrue(all(labels.count(i) == 5 for i in range(200)))

    def test_test_and_val_splits_disjoint(self):
        test_names = {b for b, _ in _read_split_resource("imagenet200_test.txt")}
        val_names = {b for b, _ in _read_split_resource("imagenet200_val.txt")}
        self.assertEqual(test_names & val_names, set())


class ImageNet200TrainTest(unittest.TestCase):
    def test_filters_and_relabels(self):
        # two in-distribution WNIDs (labels 0 and 1) plus one foreign class
        wnids = [CLASSES[0], "n99999999", CLASSES[1]]
        samples = [
            ("/in/a1.JPEG", 0),
            ("/in/a2.JPEG", 0),
            ("/foreign/b1.JPEG", 1),
            ("/in/c1.JPEG", 2),
        ]
        fake = _FakeImageNet(samples, wnids)

        with mock.patch(
            "pytorch_ood.dataset.img.imagenet200.ImageNet",
            side_effect=_fake_imagenet_factory(train=fake),
        ):
            ds = ImageNet200("/fake", split="train")

        # foreign class dropped, the three ID images kept and relabeled to 0/0/1
        self.assertEqual(len(ds), 3)
        labels = sorted(label for _, label in ds.samples)
        self.assertEqual(labels, [0, 0, 1])
        self.assertNotIn("/foreign/b1.JPEG", [p for p, _ in ds.samples])


class ImageNet200ValTestSplitTest(unittest.TestCase):
    def _fake_val_for(self, resource):
        # build a fake ImageNet val containing exactly the split's basenames,
        # plus one extra val image that must be ignored
        entries = _read_split_resource(resource)
        samples = [(f"/val/{base}", 0) for base, _ in entries]
        samples.append(("/val/ILSVRC2012_val_99999999.JPEG", 0))
        return _FakeImageNet(samples, CLASSES), entries

    def test_test_split_selection_and_labels(self):
        fake_val, entries = self._fake_val_for("imagenet200_test.txt")
        with mock.patch(
            "pytorch_ood.dataset.img.imagenet200.ImageNet",
            side_effect=_fake_imagenet_factory(val=fake_val),
        ):
            ds = ImageNet200("/fake", split="test")

        self.assertEqual(len(ds), 9000)
        # extra non-listed val image is excluded
        self.assertNotIn("/val/ILSVRC2012_val_99999999.JPEG", [p for p, _ in ds.samples])
        # labels follow the resource mapping exactly
        expected = {f"/val/{base}": lbl for base, lbl in entries}
        for path, label in ds.samples:
            self.assertEqual(label, expected[path])

    def test_val_split_size(self):
        fake_val, _ = self._fake_val_for("imagenet200_val.txt")
        with mock.patch(
            "pytorch_ood.dataset.img.imagenet200.ImageNet",
            side_effect=_fake_imagenet_factory(val=fake_val),
        ):
            ds = ImageNet200("/fake", split="val")
        self.assertEqual(len(ds), 1000)


class ImageNet200MiscTest(unittest.TestCase):
    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            ImageNet200("/fake", split="bogus")


if __name__ == "__main__":
    unittest.main()
