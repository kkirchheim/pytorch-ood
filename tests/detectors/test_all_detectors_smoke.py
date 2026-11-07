import unittest

import torch

from tests.helpers.detectors import DETECTOR_CASES, make_loader


class TestAllDetectorsSmoke(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_all_detectors_smoke(self):
        for case in DETECTOR_CASES:
            with self.subTest(detector=case.name):
                detector = case.build()
                if getattr(detector, "requires_fit", False):
                    loader = (
                        case.fit_loader() if case.fit_loader else make_loader(case.input_shape)
                    )
                    detector.fit(loader)

                scores = detector(torch.randn(8, *case.input_shape))
                self.assertIsInstance(scores, torch.Tensor)
                self.assertEqual(scores.shape, (8,))
                self.assertTrue(torch.isfinite(scores).all())
