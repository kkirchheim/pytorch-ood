import pytest

from pytorch_ood.model import ResNet18, ResNet50


@pytest.mark.parametrize("cls", [ResNet18, ResNet50])
def test_weights_argument_is_rejected(cls):
    with pytest.raises(ValueError, match="load_model"):
        cls(weights="DEFAULT")
