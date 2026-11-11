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


class TestImageNet800Resource:
    def test_classes_count_and_order(self):
        assert len(CLASSES_800) == 800
        assert len(set(CLASSES_800)) == 800
        assert CLASSES_800 == sorted(CLASSES_800)

    def test_disjoint_from_imagenet200(self):
        assert set(CLASSES_800) & set(CLASSES_200) == set()
        assert len(set(CLASSES_800) | set(CLASSES_200)) == 1000


class TestImageNet800Train:
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

        assert len(ds) == 2
        assert "/foreign/b1.JPEG" not in ds.samples

        with mock.patch("pytorch_ood.dataset.img.imagenet800.Image.open", return_value="img"):
            img, target = ds[0]
        assert target == -1
