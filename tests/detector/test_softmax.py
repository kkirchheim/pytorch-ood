import torch

from pytorch_ood.detector import MaxSoftmax


def test_temperature_is_float():
    detector = MaxSoftmax(None, t=2)

    assert isinstance(detector.t, float)
    assert detector.t == 2.0


def test_temperature_scales_logits():
    logits = torch.randn(16, 5)
    detector = MaxSoftmax(None, t=2.0)

    expected = -(logits / 2.0).softmax(dim=1).max(dim=1).values
    torch.testing.assert_close(detector.predict_logits(logits), expected)
