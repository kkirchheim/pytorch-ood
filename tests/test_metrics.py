import unittest

import torch

from src.pytorch_ood.utils import OODMetrics
from src.pytorch_ood.utils.metrics import autc_score


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
        self.assertEqual(round(r["AUTC"], 3), 0.0)
        self.assertEqual(r["AUPR-IN"], 1.0)
        self.assertEqual(r["AUPR-OUT"], 1.0)
        self.assertEqual(r["FPR95TPR"], 0.0)

    def test_autc(self):
        # split
        in_samples_num = 9
        out_samples_num = 1

        # random torch tensors
        offset = 10**3
        in_scores = torch.rand(in_samples_num * offset)
        near_out_scores = torch.rand(out_samples_num * offset) + 2
        far_out_scores = torch.rand(out_samples_num * offset) + 10

        # create labels
        labels = torch.cat([torch.zeros_like(in_scores), torch.ones_like(near_out_scores)])

        metrics_near = OODMetrics()
        metrics_near.update(torch.cat([in_scores, near_out_scores]), -labels)
        metric_dict_near = metrics_near.compute()

        metrics_far = OODMetrics()
        metrics_far.update(torch.cat([in_scores, far_out_scores]), -labels)
        metric_dict_far = metrics_far.compute()

        print("Near", metric_dict_near)
        print("Far", metric_dict_far)

        self.assertGreater(metric_dict_near["AUTC"], metric_dict_far["AUTC"])

    def test_autc_perfect_separation(self):
        # a perfect detector should have AUTC close to 0
        labels = torch.cat([torch.zeros(50), torch.ones(50)])
        scores = torch.cat([torch.zeros(50), torch.ones(50)])
        self.assertAlmostEqual(float(autc_score(labels, scores)), 0.0, places=5)

    def test_autc_matches_threshold_curve(self):
        # closed-form AUTC must match the threshold-curve definition on dense data
        from torchmetrics.functional.classification import binary_roc
        from torchmetrics.utilities.compute import auc

        torch.manual_seed(3)
        scores = torch.randn(5000)
        labels = (torch.rand(5000) > 0.5).long()

        s = (scores - scores.min()) / (scores.max() - scores.min())
        fpr, tpr, th = binary_roc(s, labels, thresholds=1000)
        fpr = torch.cat([torch.tensor([0.0]), fpr])
        tpr = torch.cat([torch.tensor([0.0]), tpr])
        th = torch.cat([torch.tensor([1.0]), th])
        reference = float((auc(th, fpr) + auc(th, 1 - tpr)) / 2)

        self.assertAlmostEqual(float(autc_score(labels, scores)), reference, places=3)

    def test_void_label(self):
        metrics = OODMetrics(void_label=2)
        scores = torch.zeros(size=(10,))
        y = torch.zeros(size=(10,))

        # add void entry
        scores[0] = -1
        y[0] = 2

        scores[5:] = 1
        y[5:] = -1
        metrics.update(scores, y)
        r = metrics.compute()
        print(r)
        self.assertEqual(r["AUROC"], 1.0)
        self.assertEqual(round(r["AUTC"], 3), 0.0)
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

    def test_buffer_honors_device(self):
        # the device parameter must control where results are buffered
        metrics = OODMetrics(device="cpu")
        scores = torch.zeros(size=(10,))
        y = torch.zeros(size=(10,))
        scores[5:] = 1
        y[5:] = -1
        metrics.update(scores, y)
        self.assertEqual(metrics.buffer.get("scores").device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_buffer_honors_device_cuda(self):
        metrics = OODMetrics(device="cuda")
        scores = torch.zeros(size=(10,), device="cuda")
        y = torch.zeros(size=(10,), device="cuda")
        scores[5:] = 1
        y[5:] = -1
        metrics.update(scores, y)
        self.assertEqual(metrics.buffer.get("scores").device.type, "cuda")
        # results must still come back as plain python floats
        r = metrics.compute()
        self.assertIsInstance(r["AUROC"], float)
        self.assertEqual(r["AUROC"], 1.0)

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
        self.assertEqual(round(r["AUTC"], 3), 0.0)
        self.assertEqual(r["AUPR-IN"], 1.0)
        self.assertEqual(r["AUPR-OUT"], 1.0)
        self.assertEqual(r["FPR95TPR"], 0.0)

    def test_segmentation3(self):
        """
        Test with unequal mask
        """
        metrics = OODMetrics(mode="segmentation")
        x = torch.zeros(size=(2, 32, 32))
        y = torch.zeros(size=(2, 48, 32))
        y[:, 1, :] = -1

        with self.assertRaises(ValueError) as context:
            metrics.update(x, y)
