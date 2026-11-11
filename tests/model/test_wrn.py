import torch

from pytorch_ood.model import WideResNet


def test_forward_is_the_classifier_on_the_features(seed: torch.Generator):
    model = WideResNet(num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32, generator=seed)

    with torch.no_grad():
        torch.testing.assert_close(model(x), model.fc(model.features(x)))
