import unittest

import torch

from src.pytorch_ood.detector import ODIN
from tests.helpers import ClassificationModel


class TestODIN(unittest.TestCase):
    """
    Test code of examples
    """

    def test_odin(self):
        model = ClassificationModel()
        odin = ODIN(model)

        x = torch.zeros(size=(1, 10))
        y = odin.predict(x)

        self.assertIsNotNone(y)
