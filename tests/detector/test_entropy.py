import math

import pytest
import torch

from pytorch_ood.detector import Entropy


@pytest.mark.parametrize("margin", [0.0, 5.0, 20.0, 30.0, 40.0])
def test_matches_closed_form(margin: float):
    # logits [m, 0, 0]: with e = exp(-m), the entropy is log(1 + 2e) + 2em / (1 + 2e)
    e = math.exp(-margin)
    expected = math.log1p(2 * e) + 2 * e * margin / (1 + 2 * e)
    scores = Entropy(None).predict_logits(torch.tensor([[margin, 0.0, 0.0]]))
    # float32 rounds the largest softmax value to 1, which loses about 1 / (m + 1) of the entropy
    torch.testing.assert_close(scores, torch.tensor([expected]), rtol=0.1, atol=0)
