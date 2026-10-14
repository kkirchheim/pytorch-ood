import pytest
import torch

from src.pytorch_ood.detector import MCD


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spatial", [False, True])
def test_mcd_mean_preserves_model_output_dtype(dtype, default_dtype, spatial):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        model = torch.nn.Identity().eval()
        logits = torch.tensor([[0.1, 0.2, 0.3], [1.1, -0.4, 0.7]], dtype=dtype)
        if spatial:
            logits = logits[:, :, None, None].expand(2, 3, 2, 4)
        expected = logits.softmax(dim=1)

        mean = MCD.run_mean(model, logits, samples=3)
        score = MCD(model, samples=3, mode="mean").predict(logits)

        assert mean.dtype == dtype
        assert score.dtype == dtype
        torch.testing.assert_close(mean, expected)
        torch.testing.assert_close(score, -expected.max(dim=1).values)
        assert not model.training
    finally:
        torch.set_default_dtype(previous_dtype)


def test_mcd_mean_matches_double_dropout_draws():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.5)).double().eval()
    inputs = torch.tensor([[0.1, 0.3, 0.7], [1.1, -0.5, 0.9]], dtype=torch.float64)
    torch.manual_seed(42)
    model.train()
    expected = torch.stack([model(inputs).softmax(dim=1) for _ in range(7)]).mean(dim=0)
    model.eval()
    torch.manual_seed(42)

    actual = MCD.run_mean(model, inputs, samples=7)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
    assert not model.training
