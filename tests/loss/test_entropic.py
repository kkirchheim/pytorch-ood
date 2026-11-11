import unittest

import pytest
import torch
import torch.nn.functional as F

from pytorch_ood.loss import EntropicOpenSetLoss
from tests.helpers.model import SegmentationModel


class TestEntropicLoss(unittest.TestCase):
    """
    Test code of examples
    """

    def test_forward(self):
        target = torch.arange(0, 128).long() % 10
        target[50:] = -1

        criterion = EntropicOpenSetLoss()
        x = torch.randn(size=(128, 10))
        loss = criterion(x, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_segmentation_with_unknown(self):
        model = SegmentationModel()
        criterion = EntropicOpenSetLoss(reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertNotEqual(loss[0, 0, 0], 0)
        loss.mean().backward()


@pytest.mark.parametrize("n_ood", [0, 6], ids=["all-id", "mixed"])
def test_cross_entropy_on_id_rows_and_uniform_target_on_ood_rows(
    n_ood: int, seed: torch.Generator
):
    logits = torch.randn(12, 4, generator=seed)
    target = torch.arange(12) % 4
    target[:n_ood] = -1
    known = target >= 0
    expected = torch.zeros(12)
    expected[known] = F.cross_entropy(logits[known], target[known], reduction="none")
    expected[~known] = -logits[~known].log_softmax(dim=1).mean(dim=1)

    torch.testing.assert_close(EntropicOpenSetLoss(reduction="none")(logits, target), expected)
