import unittest

import torch

from src.pytorch_ood.detector import LTS
from src.pytorch_ood.detector.energy import EnergyBased
from tests.helpers import ClassificationModel


class TestLTS(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ClassificationModel(num_inputs=10, n_hidden=8, num_outputs=3)
        self.model.eval()

    def test_predict_returns_correct_shape(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier, p=0.05)
        x = torch.randn(16, 10)
        scores = detector(x)
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_predict_features_returns_correct_shape(self):
        detector = LTS(encoder=None, head=self.model.classifier, p=0.05)
        features = torch.randn(16, 8)
        scores = detector.predict_features(features)
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_predict_without_encoder_raises(self):
        detector = LTS(encoder=None, head=self.model.classifier)
        x = torch.randn(4, 10)
        with self.assertRaises(Exception):
            detector.predict(x)

    def test_temperature_shape(self):
        z = torch.randn(8, 64)
        t = LTS.temperature(z, p=0.05)
        self.assertEqual(t.shape, (8,))
        self.assertTrue(torch.isfinite(t).all())

    def test_temperature_uniform_activations(self):
        """When all activations are equal, T1/T2 = n/k, temperature = (n/k)^2."""
        z = torch.ones(4, 100)
        t = LTS.temperature(z, p=0.05)
        k = max(1, round(100 * 0.05))  # 5
        expected = (100.0 / k) ** 2  # (100/5)^2 = 400
        torch.testing.assert_close(t, torch.full((4,), expected))

    def test_temperature_concentrated_activations(self):
        """When all mass is in top p%, T1 ≈ T2, temperature ≈ 1."""
        z = torch.zeros(4, 100)
        z[:, :5] = 100.0  # all mass in top 5%
        t = LTS.temperature(z, p=0.05)
        torch.testing.assert_close(t, torch.ones(4))

    def test_p_equals_one_means_no_scaling(self):
        """With p=1.0, top 100% = all activations, so T1/T2 = 1, temperature = 1."""
        z = torch.randn(4, 64).abs()
        t = LTS.temperature(z, p=1.0)
        torch.testing.assert_close(t, torch.ones(4))

    def test_custom_detector_callable(self):
        """LTS should accept any scoring function that maps logits -> scores."""
        detector = LTS(
            encoder=self.model.features,
            head=self.model.classifier,
            detector=lambda logits: -logits.max(dim=1).values,  # MaxLogit-style
        )
        x = torch.randn(8, 10)
        scores = detector(x)
        self.assertEqual(scores.shape, (8,))

    def test_does_not_require_fit(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier)
        self.assertFalse(detector.requires_fit)

    def test_scores_differ_from_plain_energy(self):
        """LTS should produce different scores than plain energy (unless S=1 everywhere)."""
        x = torch.randn(16, 10)
        with torch.no_grad():
            features = self.model.features(x)
            logits = self.model.classifier(features)

        energy_scores = EnergyBased.score(logits)
        lts_detector = LTS(encoder=self.model.features, head=self.model.classifier, p=0.05)
        lts_scores = lts_detector(x)

        self.assertFalse(torch.allclose(energy_scores, lts_scores))

    def test_batch_size_one(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier)
        x = torch.randn(1, 10)
        scores = detector(x)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_default_p(self):
        detector = LTS(encoder=self.model.features, head=self.model.classifier)
        self.assertEqual(detector.p, 0.05)

    def test_4d_feature_maps(self):
        """Temperature should handle 4D feature maps by flattening."""
        z = torch.randn(4, 16, 8, 8)
        t = LTS.temperature(z, p=0.05)
        self.assertEqual(t.shape, (4,))
        self.assertTrue(torch.isfinite(t).all())

    def test_segmentation_4d_features(self):
        """LTS should work with spatial 4D feature maps (B, C, H, W)."""
        features = torch.randn(4, 8, 16, 16)  # (B, C, H, W)
        t = LTS.temperature(features, p=0.05)
        self.assertEqual(t.shape, (4,))
        self.assertTrue(torch.isfinite(t).all())

    def test_segmentation_4d_logits_scoring(self):
        """LTS should score 4D logits (segmentation output) correctly."""
        # Simulate a conv head that outputs (B, K, H, W) logits
        batch_size, num_classes, height, width = 4, 3, 16, 16
        logits_4d = torch.randn(batch_size, num_classes, height, width)
        features_4d = torch.randn(batch_size, 8, height, width)

        # Create a simple conv head
        class ConvHead(torch.nn.Module):
            def forward(self, x):
                # x is (B, C, H, W), return (B, K, H, W)
                return torch.randn(x.shape[0], num_classes, x.shape[2], x.shape[3])

        t = LTS.temperature(features_4d, p=0.05)  # (4,)

        # Manually compute scaled logits
        scaled_logits = logits_4d / t.view(batch_size, 1, 1, 1)
        self.assertEqual(scaled_logits.shape, (batch_size, num_classes, height, width))

        # Score should work on spatial logits
        from src.pytorch_ood.detector.energy import EnergyBased

        scores = EnergyBased.score(scaled_logits)
        self.assertEqual(scores.shape, (batch_size, height, width))
        self.assertTrue(torch.isfinite(scores).all())

    def test_segmentation_predict_feature_maps(self):
        """LTS.predict_features should work with 4D spatial feature maps."""
        batch_size, num_channels, height, width = 4, 8, 16, 16
        num_classes = 3
        features_4d = torch.randn(batch_size, num_channels, height, width)

        # Create a simple conv head that outputs (B, K, H, W)
        class ConvHead(torch.nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], num_classes, x.shape[2], x.shape[3])

        detector = LTS(encoder=None, head=ConvHead(), p=0.05)
        scores = detector.predict_features(features_4d)

        # For 4D features, output should be (B, H, W)
        self.assertEqual(scores.shape, (batch_size, height, width))
        self.assertTrue(torch.isfinite(scores).all())

    def test_segmentation_full_pipeline(self):
        """LTS should work end-to-end with spatial encoder and conv head."""
        batch_size = 2
        height, width = 8, 8

        # Simple spatial encoder (e.g., backbone)
        class SpatialEncoder(torch.nn.Module):
            def forward(self, x):
                # x: (B, H, W) -> (B, 16, H, W)
                return torch.randn(x.shape[0], 16, x.shape[1], x.shape[2])

        # Simple conv head
        class ConvHead(torch.nn.Module):
            def forward(self, x):
                # x: (B, 16, H, W) -> (B, 3, H, W)
                return torch.randn(x.shape[0], 3, x.shape[2], x.shape[3])

        detector = LTS(encoder=SpatialEncoder(), head=ConvHead(), p=0.05)

        # Input: (B, H, W) spatial input
        x = torch.randn(batch_size, height, width)
        scores = detector(x)

        # Output should be (B, H, W) scores per pixel
        self.assertEqual(scores.shape, (batch_size, height, width))
        self.assertTrue(torch.isfinite(scores).all())

    def test_classification_backward_compatibility(self):
        """LTS should still work with 2D pooled features (original use case)."""
        detector = LTS(encoder=self.model.features, head=self.model.classifier, p=0.05)
        x = torch.randn(8, 10)
        scores = detector(x)
        self.assertEqual(scores.shape, (8,))
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
