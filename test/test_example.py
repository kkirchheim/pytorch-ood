import unittest
from os.path import dirname, join

example_dir = join(dirname(__file__), "..", "examples")


class TestExamples(unittest.TestCase):
    """
    Test code of examples
    """

    @unittest.skip("Not fully implemented")
    def test_example_1(self):
        import sh

        sh.python(join(example_dir, "example.py"))
