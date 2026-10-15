import unittest

import torch
from torch import nn

from src.pytorch_ood.detector import MCD
from tests.helpers import ClassificationModel, SegmentationModel


class TestMCD(unittest.TestCase):
    """
    Tests for Monte Carlo Dropout
    """

    def test_something(self):
        model = ClassificationModel()
        mcd = MCD(model, mode="var")

        x = torch.randn(size=(128, 10))
        y = mcd.predict(x)

        print(y)
        self.assertEqual(torch.Size([128]), y.shape)
        self.assertIsNotNone(y)

    def test_segmentation(self):
        model = SegmentationModel()
        mcd = MCD(model)

        x = torch.zeros(size=(4, 3, 32, 32))
        y = mcd.predict(x)

        print(y.shape)
        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (4, 32, 32))

    def test_mean_eq_mean(self):
        model = ClassificationModel()
        mcd = MCD(model, mode="mean", samples=1000)
        x = torch.randn(size=(128, 10))

        y = mcd.predict(x)

        mean = -mcd.run(model, x, samples=1000)[0]
        self.assertTrue(torch.isclose(y, mean, rtol=0.05).all())


class _FailingModel(nn.Module):
    def forward(self, x):
        raise RuntimeError("forward failed")


class TestMCDEdgeCases(unittest.TestCase):
    def test_run_mean_keeps_dtype(self):
        model = nn.Sequential(nn.Dropout(0.5), nn.Linear(10, 3)).double().eval()
        x = torch.randn(8, 10, dtype=torch.float64)

        torch.manual_seed(0)
        mean = MCD.run_mean(model, x, samples=5)

        # same dropout masks, accumulated in float64
        torch.manual_seed(0)
        model.train()
        with torch.no_grad():
            expected = torch.stack([model(x).softmax(dim=1) for _ in range(5)]).mean(dim=0)

        self.assertEqual(mean.dtype, torch.float64)
        torch.testing.assert_close(mean, expected, rtol=0, atol=1e-12)

    def test_eval_mode_restored_on_error(self):
        for run in [MCD.run, MCD.run_mean]:
            for training in [False, True]:
                with self.subTest(run=run.__name__, training=training):
                    model = _FailingModel().train(training)
                    with self.assertRaisesRegex(RuntimeError, "forward failed"):
                        run(model, torch.randn(2, 10), samples=3)
                    self.assertEqual(model.training, training)
