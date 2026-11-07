"""Hardware requirements for tests.

``@requires(cuda=1)`` marks a test that needs at least one CUDA device. Whether the hardware is
present is checked when the test is set up (see ``tests/conftest.py``), never at collection, so
collecting the suite does not initialize CUDA. Missing hardware skips the test with a reason that
starts with ``hardware:``, or fails it under ``pytest --require-hardware``.
"""

from typing import Optional

import pytest
import torch

requires = pytest.mark.hardware


def missing_hardware(cuda: int = 0, mps: bool = False) -> Optional[str]:
    """Describe what is missing to satisfy the requirement, or return ``None``."""
    if cuda and torch.cuda.device_count() < cuda:
        return f"needs cuda>={cuda}, found {torch.cuda.device_count()}"
    if mps and not torch.backends.mps.is_available():
        return "needs mps"
    return None
