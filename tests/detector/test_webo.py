import math

import pytest
import torch

from pytorch_ood.detector import WeightedEBO


@pytest.mark.parametrize(
    "logits, weights, expected",
    [
        pytest.param([0.0, math.log(3.0)], [2.0, 1.0], -math.log(5.0), id="weighted-sum"),
        # exp(100) overflows float32
        pytest.param([100.0, 0.0], [1.0, 1.0], -100.0, id="large-logit"),
        pytest.param([100.0, 0.0], [0.0, 1.0], 0.0, id="zero-weight-removes-the-class"),
        pytest.param([100.0, 0.0], [-1.0, 1.0], 0.0, id="negative-weight-is-clipped"),
    ],
)
def test_matches_closed_form(logits, weights, expected):
    scores = WeightedEBO(None, torch.tensor(weights)).predict_logits(torch.tensor([logits]))
    torch.testing.assert_close(scores, torch.tensor([expected]))
