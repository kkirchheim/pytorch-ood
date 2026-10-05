import unittest

import torch

from src.pytorch_ood.detector import ASH
from src.pytorch_ood.model import WideResNet


class TestASH(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def test_input(self):
        """ """
        model = WideResNet(num_classes=10).eval()
        detector = ASH(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )

        x = torch.randn(size=(16, 3, 32, 32))

        output = detector(x)

        print(output)
        self.assertIsNotNone(output)


def _old_in_place(x, percentile, variant):
    """The previous in-place implementation, as reference for the values."""
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    if variant == "ash-b":
        v = (s1 / k).unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=v)
    if variant == "ash-s":
        x = x * torch.exp((s1 / x.sum(dim=[1, 2, 3]))[:, None, None, None])
    return x


class TestASHOutOfPlace(unittest.TestCase):
    variants = ["ash-p", "ash-b", "ash-s"]

    @staticmethod
    def _detector(variant):
        return ASH(backbone=None, head=lambda f: f.mean(dim=(2, 3)), variant=variant)

    def test_input_unchanged(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                x = torch.rand(4, 8, 5, 5)
                before = x.clone()
                self._detector(variant).predict_feature_maps(x)
                self.assertTrue(torch.equal(x, before))

    def test_non_contiguous(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                x = torch.rand(4, 5, 5, 8).permute(0, 3, 1, 2)
                self.assertFalse(x.is_contiguous())
                detector = self._detector(variant)
                torch.testing.assert_close(
                    detector.predict_feature_maps(x),
                    detector.predict_feature_maps(x.contiguous()),
                )

    def test_same_values_as_in_place_version(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                torch.manual_seed(0)
                x = torch.rand(4, 8, 5, 5)
                detector = self._detector(variant)
                torch.testing.assert_close(
                    detector.ash(x, 0.65), _old_in_place(x.clone(), 0.65, variant)
                )
