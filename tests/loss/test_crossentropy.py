import pytest
import torch
import torch.nn.functional as F

from pytorch_ood.loss import CrossEntropyLoss


def test_segmentation_mean_is_over_id_pixels():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 4)
    target = torch.randint(0, 3, (2, 4, 4))
    target[0, :2] = -1
    known = target >= 0

    with pytest.warns(UserWarning):
        value = CrossEntropyLoss()(logits, target)

    expected = F.cross_entropy(logits.permute(0, 2, 3, 1)[known], target[known])
    torch.testing.assert_close(value, expected)
