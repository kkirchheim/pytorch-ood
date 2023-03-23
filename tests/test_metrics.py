import unittest

import torch

from src.pytorch_ood.utils import OODMetrics


class TestMetrics(unittest.TestCase):
    """
    Test calculation of metrics
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
        self.assertEqual(r["FPR95TPR"], 0.0)

    def test_error_on_only_IN_data(self):
        metrics = OODMetrics()
        scores = torch.zeros(size=(10,))
        y = torch.zeros(size=(10,))
        scores[5:] = 1
        metrics.update(scores, y)

        with self.assertRaises(ValueError):
            r = metrics.compute()

    def test_error_on_only_OOD_data(self):
        metrics = OODMetrics()
        scores = torch.zeros(size=(10,))
        y = -1 * torch.zeros(size=(10,))
        metrics.update(scores, y)

        with self.assertRaises(ValueError):
            r = metrics.compute()

    def test_reset_1(self):
        metrics = OODMetrics()
        metrics.reset()

    def test_segmentation1(self):
        metrics = OODMetrics(mode="segmentation")
        x = torch.zeros(size=(2, 32, 32))
        y = torch.zeros(size=(2, 32, 32))
        y[:, 1, :] = -1

        metrics.update(x, y)
        metrics.compute()

    def test_segmentation2(self):
        metrics = OODMetrics(mode="segmentation")
        x = torch.zeros(size=(2, 32, 32))
        y = torch.zeros(size=(2, 32, 32))
        y[:, 1, :] = -1
        x[:, 1, :] = 1

        metrics.update(x, y)
        r = metrics.compute()
        self.assertEqual(r["AUROC"], 1.0)
        self.assertEqual(r["AUPR-IN"], 1.0)
        self.assertEqual(r["AUPR-OUT"], 1.0)
        self.assertEqual(r["FPR95TPR"], 0.0)
