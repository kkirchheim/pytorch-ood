import pytest
import torch

from src.pytorch_ood.detector import ASH, EnergyBased
from src.pytorch_ood.detector.ash import ash_s


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("percentile", [0.0, 0.65, 1.0])
def test_ash_s_zero_maps_are_finite(dtype, percentile):
    features = torch.zeros(2, 3, 2, 4, dtype=dtype, requires_grad=True)
    actual = ash_s(features, percentile)
    torch.testing.assert_close(actual, torch.zeros_like(features))
    actual.sum().backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_ash_s_fully_pruned_maps_are_zero(dtype):
    features = torch.full((2, 3, 2, 4), 1000.0, dtype=dtype, requires_grad=True)
    original = features.detach().clone()
    actual = ash_s(features, 1.0)
    torch.testing.assert_close(actual, torch.zeros_like(features))
    torch.testing.assert_close(features, original)
    actual.sum().backward()
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_ash_s_zero_sample_does_not_change_nonzero_sample(dtype):
    features = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]]], [[[1.0, 2.0, 3.0, 4.0]]]], dtype=dtype)
    expected = torch.zeros_like(features)
    expected[1, 0, 0, 2:] = torch.tensor([3.0, 4.0], dtype=dtype) * torch.exp(
        torch.tensor(10.0 / 7.0, dtype=dtype)
    )
    torch.testing.assert_close(ash_s(features, 0.5), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ash_s_scores_blank_relu_backbone(dtype):
    convolution = torch.nn.Conv2d(1, 2, 1).to(dtype)
    with torch.no_grad():
        convolution.weight.fill_(1.0)
        convolution.bias.fill_(-1.0)
    backbone = torch.nn.Sequential(convolution, torch.nn.ReLU()).eval()
    head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    detector = ASH(backbone, head, variant="ash-s")
    expected = EnergyBased.score(torch.zeros(2, 2, dtype=dtype))
    torch.testing.assert_close(detector(torch.zeros(2, 1, 4, 5, dtype=dtype)), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_ash_s_nonzero_analytic_reference(dtype):
    features = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=dtype)
    expected = torch.zeros_like(features)
    expected[0, 0, 0, 2:] = features[0, 0, 0, 2:] * torch.exp(
        torch.tensor(10.0 / 7.0, dtype=dtype)
    )
    torch.testing.assert_close(ash_s(features, 0.5), expected)
