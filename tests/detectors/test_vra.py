import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import VRA
from tests.helpers import ClassificationModel, sample_dataset


class TestVRA(unittest.TestCase):
    """
    Tests for VRA detector
    """

    def setUp(self):
        self.model = ClassificationModel(num_inputs=2, n_hidden=10, num_outputs=3).eval()
        self.dataset = sample_dataset(n_samples=50, n_dim=2)

    def test_fit_and_predict(self):
        detector = VRA(
            backbone=self.model.features,
            head=self.model.classifier,
        )
        loader = DataLoader(self.dataset, batch_size=16)
        detector.fit(loader)

        x = torch.randn(8, 2)
        scores = detector(x)
        self.assertEqual(scores.shape, (8,))

    def test_fit_and_predict_with_spatial_feature_maps(self):
        """
        Regression test: fit() must preserve the (N, C, H, W) shape of feature maps
        from a real conv backbone (H, W > 1), since predict_feature_maps() clips
        un-flattened feature maps of that same shape.
        """
        backbone = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3)
        )
        detector = VRA(backbone=backbone, head=head)

        x = torch.randn(20, 3, 8, 8)
        y = torch.zeros(20, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        detector.fit(loader)

        scores = detector(torch.randn(5, 3, 8, 8))
        self.assertEqual(scores.shape, (5,))

    def test_fit_feature_maps_and_predict_feature_maps(self):
        detector = VRA(
            backbone=self.model.features,
            head=self.model.classifier,
        )

        x, y = self.dataset.tensors
        with torch.no_grad():
            z = self.model.features(x)

        detector.fit_feature_maps(z, y)
        scores = detector.predict_feature_maps(z[:8])
        self.assertEqual(scores.shape, (8,))

    def test_requires_fitting(self):
        detector = VRA(
            backbone=self.model.features,
            head=self.model.classifier,
        )
        with self.assertRaises(Exception):
            detector(torch.randn(8, 2))

    def test_custom_percentiles(self):
        detector = VRA(
            backbone=self.model.features,
            head=self.model.classifier,
            lower_percentile=5.0,
            upper_percentile=95.0,
        )
        loader = DataLoader(self.dataset, batch_size=16)
        detector.fit(loader)

        x = torch.randn(8, 2)
        scores = detector(x)
        self.assertEqual(scores.shape, (8,))
