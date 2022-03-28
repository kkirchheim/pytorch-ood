import unittest
from os.path import dirname, join

import sh

example_dir = join(dirname(__file__), "..", "examples")


class TestExamples(unittest.TestCase):
    """
    Test code of examples
    """

    @unittest.skip("Not fully implemented")
    def test_example_1(self):
        sh.python(join(example_dir, "example.py"))
