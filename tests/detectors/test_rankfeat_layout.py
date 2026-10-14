import pytest
import torch

from src.pytorch_ood.detector import EnergyBased, RankFeat
from src.pytorch_ood.detector.rankfeat import _remove_rank1


def feature_layout(x, layout):
    if layout == "transpose":
        return x.transpose(-1, -2)
    if layout == "strided":
        return x[:, :, ::2, ::2]
    if layout == "channels_last":
        return x.contiguous(memory_format=torch.channels_last)
    return x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["transpose", "strided", "channels_last", "contiguous"])
def test_remove_rank1_feature_layout(dtype, layout):
    generator = torch.Generator().manual_seed(42)
    x = feature_layout(torch.randn(2, 3, 6, 8, generator=generator, dtype=dtype), layout)
    original = x.clone()
    matrix = x.contiguous().flatten(2)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    expected = matrix - s[:, :1, None] * (u[:, :, :1] @ vh[:, :1, :])

    actual = _remove_rank1(x)

    torch.testing.assert_close(actual, expected.reshape_as(x))
    torch.testing.assert_close(x, original)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["transpose", "strided", "channels_last", "contiguous"])
def test_rankfeat_predict_backbone_layout(dtype, layout):
    torch.manual_seed(42)
    model = torch.nn.Conv2d(2, 3, 1).to(dtype)
    head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    x = torch.randn(2, 2, 6, 8, dtype=dtype)
    detector = RankFeat(lambda inputs: feature_layout(model(inputs), layout), head)
    with torch.no_grad():
        features = feature_layout(model(x), layout)
        expected = EnergyBased.score(head(_remove_rank1(features.contiguous())))

    torch.testing.assert_close(detector(x), expected)
    torch.testing.assert_close(detector.predict_feature_maps(features), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_remove_rank1_transposed_gradient(dtype):
    torch.manual_seed(42)
    inputs = torch.randn(2, 3, 4, 5, dtype=dtype, requires_grad=True)
    reference = inputs.detach().clone().requires_grad_()
    actual = _remove_rank1(inputs.transpose(-1, -2))
    expected = _remove_rank1(reference.transpose(-1, -2).contiguous())
    weights = torch.arange(actual.numel(), dtype=dtype).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs.grad, reference.grad)
    assert torch.isfinite(inputs.grad).all()
