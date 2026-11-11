import pytest
import torch

from pytorch_ood.model import list_models, load_model
from pytorch_ood.model.registry import get_model_info


@pytest.mark.download
@pytest.mark.parametrize("key", list_models())
def test_every_registered_model_loads_and_gives_outputs_of_the_right_shape(key: str):
    entry = get_model_info(key)
    model = load_model(key)
    assert not model.training

    y = model(torch.ones(size=(1, 3, 32, 32)))

    assert y.shape == (1, entry.arch_kwargs["num_classes"])
