"""The tests must exercise the checkout they live in.

``pythonpath = src .`` in ``setup.cfg`` puts this checkout's ``src`` ahead of any installed copy, so
running pytest in a git worktree tests the worktree and not the main checkout behind an editable
install.
"""

from pathlib import Path

import pytorch_ood

SRC = Path(__file__).resolve().parents[2] / "src"


def test_pytorch_ood_is_imported_from_this_checkout():
    assert SRC in Path(pytorch_ood.__file__).resolve().parents
