"""
Datasets that list files must sort the listings: ``os.listdir`` and ``glob`` return them in the
order of the file system, which differs between machines, so sample indices (and with them
subsets, splits and seeded sampling) would not be reproducible.
"""

import ast
import unittest
from pathlib import Path

import pytorch_ood.dataset

DATASET_DIR = Path(pytorch_ood.dataset.__file__).parent

# calls whose order depends on the file system, by the name they are called with
LISTINGS = {"listdir", "scandir", "glob", "iglob", "glb", "iterdir", "walk"}


def _name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def unsorted_listings(source: str):
    """Line numbers of file listings that are not the direct argument of ``sorted()``."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node) in LISTINGS:
            parent = parents.get(node)
            if not (isinstance(parent, ast.Call) and _name(parent) == "sorted"):
                yield node.lineno


class TestFileOrder(unittest.TestCase):
    def test_file_listings_are_sorted(self):
        files = sorted(DATASET_DIR.rglob("*.py"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(str(path.relative_to(DATASET_DIR))):
                lines = sorted(unsorted_listings(path.read_text()))
                self.assertEqual(lines, [], "wrap the listing in sorted()")

    def test_detects_unsorted_listings(self):
        source = "import os\na = os.listdir('x')\nb = sorted(os.listdir('x'))\nc = glob('*')\n"
        self.assertEqual(sorted(unsorted_listings(source)), [2, 4])
