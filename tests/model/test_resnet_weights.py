import pytest
import torch
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from pytorch_ood.model import ResNet18, ResNet50


@pytest.mark.parametrize(
    "cls,weights_enum", [(ResNet18, ResNet18_Weights), (ResNet50, ResNet50_Weights)]
)
def test_weights_are_loaded(cls, weights_enum, monkeypatch):
    torch.manual_seed(0)
    expected = cls().state_dict()
    requested = []

    def get_state_dict(self, *args, **kwargs):
        requested.append(self)
        return expected

    monkeypatch.setattr(weights_enum, "get_state_dict", get_state_dict)

    torch.manual_seed(1)
    model = cls(weights="DEFAULT")

    assert requested == [weights_enum.DEFAULT]
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key]), key


@pytest.mark.parametrize("cls", [ResNet18, ResNet50])
def test_num_classes_conflicting_with_weights_raises(cls):
    with pytest.raises(ValueError, match="num_classes=10"):
        cls(num_classes=10, weights="DEFAULT")
