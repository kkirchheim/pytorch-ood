import pytest
import torch

from pytorch_ood.utils import drop_unknown


def test_drop_unknown_rejects_non_vector_targets():
    with pytest.raises(ValueError):
        drop_unknown(torch.zeros(2, 4, 4).long(), torch.zeros(2, 3, 4, 4))
