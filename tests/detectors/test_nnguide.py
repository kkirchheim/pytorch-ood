import unittest

import torch
from torch.utils.data import DataLoader

from pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import NNGuide
from tests.helpers import ClassificationModel, sample_dataset


class TestNNGuide(unittest.TestCase):
    """
    Tests for NNGuide Detector
    """

    def _make_detector(self):
        model = ClassificationModel(num_inputs=10, n_hidden=10, num_outputs=3)
        model.eval()
        w = model.classifier.weight.detach()
        b = model.classifier.bias.detach()
        backbone = model.features
        return NNGuide(backbone, w=w, b=b, k=5)

    def test_requires_fitting(self):
        detector = self._make_detector()
        z = torch.randn(16, 10)
        with self.assertRaises(RequiresFittingException):
            detector.predict_features(z)

    def test_fit_and_predict_features(self):
        detector = self._make_detector()
        ds = sample_dataset(n_dim=10)
        loader = DataLoader(ds, batch_size=32)

        detector.fit(loader)

        z = torch.randn(16, 10)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fit_features_direct(self):
        detector = self._make_detector()
        features = torch.randn(100, 10)
        labels = torch.zeros(100, dtype=torch.long)

        detector.fit_features(features, labels)

        z = torch.randn(8, 10)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (8,))

    def test_ignores_ood_in_fit(self):
        detector = self._make_detector()
        features = torch.randn(100, 10)
        labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.full((50,), -1)])

        detector.fit_features(features, labels)

        # only 50 ID samples should be in the bank
        self.assertEqual(detector._scaled_features.shape[0], 50)
