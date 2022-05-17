import unittest
from test.helpers import ClassificationModel

import torch

from pytorch_ood import ODIN


class TestExamples(unittest.TestCase):
    """
    Test code of examples
    """

    def test_odin(self):
        model = ClassificationModel()
        odin = ODIN(model)

        x = torch.zeros(size=(1, 10))
        y = odin.predict(x)

        self.assertIsNotNone(y)
