import unittest
from unittest import mock

from pytorch_ood.dataset.img import ImageNet800
from pytorch_ood.dataset.img.imagenet200 import _load_classes as _load_classes_200
from pytorch_ood.dataset.img.imagenet800 import _load_classes as _load_classes_800


class _FakeImageNet:
    """Stand-in for torchvision.datasets.ImageNet exposing .samples and .wnids."""

    def __init__(self, samples, wnids):
        self.samples = samples
        self.wnids = wnids


CLASSES_800 = _load_classes_800()
CLASSES_200 = _load_classes_200()


class ImageNet800ResourceTest(unittest.TestCase):
    def test_classes_count_and_order(self):
        self.assertEqual(len(CLASSES_800), 800)
        self.assertEqual(len(set(CLASSES_800)), 800)
        self.assertEqual(CLASSES_800, sorted(CLASSES_800))

    def test_disjoint_from_imagenet200(self):
        self.assertEqual(set(CLASSES_800) & set(CLASSES_200), set())
        self.assertEqual(len(set(CLASSES_800) | set(CLASSES_200)), 1000)


class ImageNet800TrainTest(unittest.TestCase):
    def test_filters_and_labels_unknown(self):
        # one ImageNet-800 WNID, one ImageNet-200 (foreign, must be dropped) WNID
        wnids = [CLASSES_800[0], CLASSES_200[0]]
        samples = [
            ("/out/a1.JPEG", 0),
            ("/out/a2.JPEG", 0),
            ("/foreign/b1.JPEG", 1),
        ]
        fake = _FakeImageNet(samples, wnids)

        with mock.patch(
            "pytorch_ood.dataset.img.imagenet800.ImageNet",
            return_value=fake,
        ):
            ds = ImageNet800("/fake")

        self.assertEqual(len(ds), 2)
        self.assertNotIn("/foreign/b1.JPEG", ds.samples)

        with mock.patch("pytorch_ood.dataset.img.imagenet800.Image.open", return_value="img"):
            img, target = ds[0]
        self.assertEqual(target, -1)


if __name__ == "__main__":
    unittest.main()
