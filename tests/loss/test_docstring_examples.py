"""
Runs the usage examples (``.. code-block:: python``) in the docstrings of the losses, so that
they do not go stale.
"""

import textwrap
import unittest
import warnings

import torch

from pytorch_ood import loss
from tests.test_info import LOSSES

# losses whose contract differs from loss(logits, targets), so they need a usage example
NEED_EXAMPLE = {
    loss.CACLoss,
    loss.CenterLoss,
    loss.ConfidenceLoss,
    loss.DeepSADLoss,
    loss.DeepSVDDLoss,
    loss.EnergyMarginLoss,
    loss.IILoss,
    loss.MCHADLoss,
    loss.ObjectosphereLoss,
    loss.VOSRegLoss,
    loss.VirtualOutlierSynthesizingRegLoss,
}


def code_blocks(doc: str):
    """The content of every ``.. code-block:: python`` directive in a docstring."""
    lines = (doc or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        indent = len(line) - len(line.lstrip())
        block = []
        for body in lines[i + 1 :]:
            if body.strip() and len(body) - len(body.lstrip()) <= indent:
                break
            block.append(body)
        yield textwrap.dedent("\n".join(block)).strip()


class TestLossDocstringExamples(unittest.TestCase):
    def test_losses_with_a_different_contract_have_an_example(self):
        for cls in NEED_EXAMPLE:
            with self.subTest(cls.__name__):
                self.assertTrue(list(code_blocks(cls.__doc__)), "no usage example")

    def test_examples_run(self):
        for cls in LOSSES:
            for i, code in enumerate(code_blocks(cls.__doc__)):
                with self.subTest(f"{cls.__name__}, example {i}"):
                    torch.manual_seed(0)
                    with warnings.catch_warnings():
                        # unsupervised losses warn about the OOD samples they discard
                        warnings.simplefilter("ignore", UserWarning)
                        exec(compile(code, f"<{cls.__name__} example {i}>", "exec"), {})
