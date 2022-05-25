import unittest
from os.path import dirname, join

import torch

from src.pytorch_ood import utils

example_dir = join(dirname(__file__), "..", "examples")


class TestUtils(unittest.TestCase):
    """
    Test code of examples
    """

    def test_callibration_error(self):
        conf = torch.linspace(0, 1, 1000)
        y = torch.ones(
            1000,
        )
        y[500:] = 0

        print(conf.shape)
        utils.calibration_error(conf, y)

    def test_openness(self):
        utils.calc_openness(n_train=6, n_test=10, n_target=6)
