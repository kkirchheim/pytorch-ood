import pytest
import torch

from src.pytorch_ood.detector import MCD


@pytest.mark.parametrize("mode", ["mean", "var"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fail_on", [1, 2])
@pytest.mark.parametrize("public_predict", [False, True])
def test_mcd_restores_mode_after_forward_error(mode, training, fail_on, public_predict):
    error = RuntimeError("backbone failed")

    class FailingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.dropout = torch.nn.Dropout()
            self.linear = torch.nn.Linear(3, 2)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            assert self.training
            logits = self.linear(self.dropout(self.bn(x)))
            if self.calls == fail_on:
                raise error
            return logits

    model = FailingModel().train(training)
    inputs = torch.ones(2, 3)
    original_modes = [module.training for module in model.modules()]
    mean = model.bn.running_mean.clone()
    variance = model.bn.running_var.clone()
    with pytest.raises(RuntimeError, match="backbone failed") as caught:
        if public_predict:
            MCD(model, samples=3, mode=mode).predict(inputs)
        elif mode == "mean":
            MCD.run_mean(model, inputs, samples=3)
        else:
            MCD.run(model, inputs, samples=3)

    assert caught.value is error
    assert model.calls == fail_on
    assert [module.training for module in model.modules()] == original_modes
    if not training:
        torch.testing.assert_close(model.bn.running_mean, mean)
        torch.testing.assert_close(model.bn.running_var, variance)
