import torch

from pytorch_ood.detector import MaxSoftmax
from tests.helpers import ClassificationModel, requires


class TestSoftmax:
    """
    Tests for Energy based Out-of-Distribution Detection
    """

    def test_temperature_is_float(self):
        detector = MaxSoftmax(None, t=2)

        assert isinstance(detector.t, float)
        assert detector.t == 2.0

    def test_temperature_scales_logits(self):
        logits = torch.randn(16, 5)
        detector = MaxSoftmax(None, t=2.0)

        expected = -(logits / 2.0).softmax(dim=1).max(dim=1).values
        torch.testing.assert_close(detector.predict_logits(logits), expected)

    @requires(cuda=1)
    def test_cuda_model_without_to(self):
        # the temperature must not pin the detector to the CPU when the model is on the GPU
        model = ClassificationModel().cuda()
        detector = MaxSoftmax(model)

        x = torch.zeros(size=(8, 10), device="cuda")
        with torch.no_grad():
            y = detector(x)
        assert y.device == x.device
        assert y.shape == (8,)
