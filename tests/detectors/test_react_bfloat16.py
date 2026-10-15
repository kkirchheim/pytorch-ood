import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import ReAct


class TestReActBfloat16(unittest.TestCase):
    def test_threshold_from_cnn_activations(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            for use_loader in (False, True):
                with self.subTest(dtype=dtype, loader=use_loader):
                    backbone = (
                        torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1, bias=False), torch.nn.ReLU())
                        .to(dtype)
                        .eval()
                    )
                    with torch.no_grad():
                        backbone[0].weight.copy_(
                            torch.tensor([1.0, 2.0], dtype=dtype).reshape(2, 1, 1, 1)
                        )
                    head = torch.nn.Sequential(
                        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
                    ).to(dtype)
                    x = torch.arange(16, dtype=dtype).reshape(4, 1, 2, 2)
                    x[2] = 1000
                    labels = torch.tensor([0, 1, -1, 0])
                    original = x.clone()
                    with torch.no_grad():
                        features = backbone(x)
                    original_features = features.clone()
                    ordered = features[labels >= 0].double().flatten().sort().values
                    middle = len(ordered) // 2
                    expected_threshold = (ordered[middle - 1] + ordered[middle]).item() / 2
                    detector = ReAct(backbone, head, percentile=0.5)
                    if use_loader:
                        detector.fit(DataLoader(TensorDataset(x, labels), batch_size=2))
                    else:
                        detector.fit_feature_maps(features, labels)
                    self.assertEqual(detector.threshold, expected_threshold)
                    scores = detector.predict(x)
                    with torch.no_grad():
                        expected = -torch.logsumexp(
                            head(features.clamp(max=expected_threshold)), dim=1
                        )
                    torch.testing.assert_close(scores, expected)
                    self.assertTrue(torch.isfinite(scores).all())
                    torch.testing.assert_close(x, original)
                    torch.testing.assert_close(features, original_features)
