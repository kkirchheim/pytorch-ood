"""Markers for tests that need PyTorch features newer than the oldest supported release.

``@requires_torch_func`` skips a test when ``torch.func`` (PyTorch >= 2.0) is missing, e.g. in the
CI job that installs the minimum supported dependency versions.
"""

import pytest

try:
    import torch.func  # noqa: F401

    HAS_TORCH_FUNC = True
except ImportError:
    HAS_TORCH_FUNC = False

requires_torch_func = pytest.mark.skipif(
    not HAS_TORCH_FUNC, reason="needs torch.func (torch>=2.0)"
)
