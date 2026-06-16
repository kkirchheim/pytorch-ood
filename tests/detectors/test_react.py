import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import ReAct
from src.pytorch_ood.model import WideResNet
from tests.helpers import sample_dataset


class TestReAct(unittest.TestCase):
    """
    Tests for ReAct
    """

    def test_input_after_fit(self):
        """Threshold is estimated from training data, then scoring works."""
        model = WideResNet(num_classes=10).eval()
        detector = ReAct(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )

        x_train = torch.randn(size=(8, 3, 32, 32))
        y_train = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
        detector.fit(loader)

        self.assertIsNotNone(detector.threshold)

        output = detector(torch.randn(size=(16, 3, 32, 32)))
        self.assertIsNotNone(output)

    def test_explicit_threshold_needs_no_fit(self):
        """Passing an explicit threshold allows scoring without fitting."""
        model = WideResNet(num_classes=10).eval()
        detector = ReAct(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
            threshold=1.0,
        )
        output = detector(torch.randn(size=(8, 3, 32, 32)))
        self.assertIsNotNone(output)

    def test_predict_without_threshold_raises(self):
        """Without a threshold or fit, prediction must fail explicitly."""
        model = WideResNet(num_classes=10).eval()
        detector = ReAct(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
        )
        with self.assertRaises(RequiresFittingException):
            detector(torch.randn(size=(8, 3, 32, 32)))

    def test_threshold_matches_percentile(self):
        """fit_feature_maps sets the threshold to the requested activation percentile."""
        feature_maps = torch.arange(0, 100, dtype=torch.float32).reshape(10, 10)
        y = torch.zeros(10, dtype=torch.long)

        detector = ReAct(backbone=None, head=lambda x: x, percentile=0.9)
        detector.fit_feature_maps(feature_maps, y)

        expected = float(np.percentile(feature_maps.numpy(), 90))
        self.assertAlmostEqual(detector.threshold, expected, places=4)

    def test_ood_samples_ignored_in_fit(self):
        """OOD samples in the loader must not break fitting."""
        ds = sample_dataset(n_dim=8, seed=5)
        loader = DataLoader(ds, batch_size=32)

        detector = ReAct(backbone=lambda x: x, head=lambda x: x, percentile=0.9)
        detector.fit(loader)
        self.assertIsNotNone(detector.threshold)


if __name__ == "__main__":
    unittest.main()
