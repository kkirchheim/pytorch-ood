"""
Value tests for :class:`TemperatureScaling`; the detector contract covers its API.
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor

from pytorch_ood.detector import TemperatureScaling


def _calibrated_after_dividing_by(t: float, n: int = 4000, c: int = 5) -> Tuple[Tensor, Tensor]:
    """Logits ``t * z`` with labels drawn from ``softmax(z)``: dividing by ``t`` calibrates them."""
    generator = torch.Generator().manual_seed(0)
    z = 3 * torch.randn(n, c, generator=generator)
    y = torch.multinomial(z.softmax(dim=1), 1, generator=generator).squeeze(1)
    return t * z, y


@pytest.mark.parametrize("t", [0.5, 4.0])
def test_fit_recovers_the_calibrating_temperature(t: float):
    logits, y = _calibrated_after_dividing_by(t)

    detector = TemperatureScaling(None).fit_logits(logits, y)

    assert detector.t.item() == pytest.approx(t, rel=0.05)
