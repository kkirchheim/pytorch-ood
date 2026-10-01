import unittest

import torch

from src.pytorch_ood.detector import ASH, DICE, LTS, SCALE, VRA, ReAct


def _identity(x):
    return x


# detectors whose percentile/fraction hyperparameters must lie in [0, 1]
BUILDERS = {
    "ASH.percentile": lambda v: ASH(_identity, _identity, percentile=v),
    "SCALE.percentile": lambda v: SCALE(_identity, _identity, percentile=v),
    "ReAct.percentile": lambda v: ReAct(_identity, _identity, percentile=v),
    "LTS.p": lambda v: LTS(_identity, torch.nn.Identity(), p=v),
    "DICE.p": lambda v: DICE(_identity, torch.zeros(3, 4), torch.zeros(3), p=v),
    "VRA.lower_percentile": lambda v: VRA(_identity, _identity, lower_percentile=v),
    "VRA.upper_percentile": lambda v: VRA(_identity, _identity, upper_percentile=v),
}


class TestFractionHyperparameters(unittest.TestCase):
    """Percentiles are fractions in [0, 1] throughout the library."""

    def test_accepts_fractions(self):
        for name, build in BUILDERS.items():
            with self.subTest(name):
                build(0.5 if "upper" not in name else 0.99)

    def test_rejects_percent(self):
        for name, build in BUILDERS.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, r"use 0\.95 instead of 95"):
                    build(95.0)

    def test_rejects_negative(self):
        for name, build in BUILDERS.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    build(-0.1)

    def test_vra_lower_above_upper(self):
        with self.assertRaises(ValueError):
            VRA(_identity, _identity, lower_percentile=0.9, upper_percentile=0.5)
