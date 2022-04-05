import unittest

import torch

from oodtk.metrics import OODMetrics


class TestMetrics(unittest.TestCase):
    """
    Test code of examples
    """

    def test_example_1(self):
        metrics = OODMetrics()
        scores = torch.zeros(size=(10,))
        y = torch.zeros(size=(10,))
        scores[5:] = 1
        y[5:] = -1
        metrics.update(scores, y)
        r = metrics.compute()
        print(r)
        self.assertEqual(r["AUROC"], 1.0)
        self.assertEqual(r["AUPR-IN"], 1.0)
        self.assertEqual(r["AUPR-OUT"], 1.0)
        self.assertEqual(r["ACC95TPR"], 1.0)
        self.assertEqual(r["FPR95TPR"], 0.0)
