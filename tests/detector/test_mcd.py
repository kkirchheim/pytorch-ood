import pytest
import torch
from torch import nn

from pytorch_ood.detector import MCD
from tests.helpers import ClassificationModel


def test_mean_eq_mean():
    model = ClassificationModel()
    mcd = MCD(model, mode="mean", samples=1000)
    x = torch.randn(size=(128, 10))

    y = mcd.predict(x)

    mean = -mcd.run(model, x, samples=1000)[0]
    assert torch.isclose(y, mean, rtol=0.05).all()


class _FailingModel(nn.Module):
    def forward(self, x):
        raise RuntimeError("forward failed")


def test_run_mean_keeps_dtype():
    model = nn.Sequential(nn.Dropout(0.5), nn.Linear(10, 3)).double().eval()
    x = torch.randn(8, 10, dtype=torch.float64)

    torch.manual_seed(0)
    mean = MCD.run_mean(model, x, samples=5)

    # same dropout masks, accumulated in float64
    torch.manual_seed(0)
    model.train()
    with torch.no_grad():
        expected = torch.stack([model(x).softmax(dim=1) for _ in range(5)]).mean(dim=0)

    assert mean.dtype == torch.float64
    torch.testing.assert_close(mean, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("run", [MCD.run, MCD.run_mean])
@pytest.mark.parametrize("training", [False, True])
def test_eval_mode_restored_on_error(run, training):
    model = _FailingModel().train(training)
    with pytest.raises(RuntimeError, match="forward failed"):
        run(model, torch.randn(2, 10), samples=3)
    assert model.training == training
