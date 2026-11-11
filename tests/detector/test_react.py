import numpy as np
import pytest
import torch

from pytorch_ood.detector import ReAct
from pytorch_ood.model import WideResNet


class TestReAct:
    """
    Tests for ReAct
    """

    def test_explicit_threshold_needs_no_fit(self):
        """Passing an explicit threshold allows scoring without fitting."""
        model = WideResNet(num_classes=10).eval()
        detector = ReAct(
            backbone=model.feature_maps,
            head=model.forward_feature_maps,
            threshold=1.0,
        )
        output = detector(torch.randn(size=(8, 3, 32, 32)))
        assert output is not None

    def test_threshold_matches_percentile(self):
        """fit_feature_maps sets the threshold to the requested activation percentile."""
        feature_maps = torch.arange(0, 100, dtype=torch.float32).reshape(10, 10)
        y = torch.zeros(10, dtype=torch.long)

        detector = ReAct(backbone=None, head=lambda x: x, percentile=0.9)
        detector.fit_feature_maps(feature_maps, y)

        expected = float(np.percentile(feature_maps.numpy(), 90))
        assert detector.threshold == pytest.approx(expected, abs=5e-5)
