import unittest


class TestDatasets(unittest.TestCase):
    """
    Test datasets
    """

    @unittest.skip("Skip downloading large datasets")
    def test_something(self):
        self.assertEqual(True, True)
