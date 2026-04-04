import unittest

import torch

from src.pytorch_ood.detector import GMM, MaxLogit, MaxSoftmax
from tests.helpers import ClassificationModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device handling tests")
class TestDetectorDeviceHandling(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.device = torch.device("cuda:0")

    def test_to_moves_detector_state(self):
        model = ClassificationModel()
        detector = MaxSoftmax(model)

        detector.to(self.device)

        self.assertEqual(detector.device, self.device)
        self.assertEqual(next(detector.model.parameters()).device, self.device)
        self.assertEqual(detector.t.device, self.device)

    def test_logits_predict_methods_accept_cpu_inputs_for_cuda_detector(self):
        model = ClassificationModel().to(self.device)
        detector = MaxLogit(model).to(self.device)

        raw_inputs = torch.randn(8, 10)
        scores = detector(raw_inputs)

        self.assertEqual(scores.device, self.device)
        self.assertEqual(scores.shape, (8,))

        cached_logits = torch.randn(8, 3)
        cached_scores = detector.predict_logits(cached_logits)

        self.assertEqual(cached_scores.device, self.device)
        self.assertEqual(cached_scores.shape, (8,))

    def test_feature_predict_methods_accept_cpu_inputs_for_cuda_detector(self):
        detector = GMM(model=None)

        features = torch.randn(20, 10)
        labels = torch.cat([torch.zeros(10), torch.ones(10)])
        detector.fit_features(features, labels)
        detector.to(self.device)

        self.assertEqual(detector.device, self.device)
        self.assertEqual(detector._mu.device, self.device)
        self.assertEqual(detector._precision.device, self.device)

        eval_features = torch.randn(6, 10)
        scores = detector.predict_features(eval_features)

        self.assertEqual(scores.device, self.device)
        self.assertEqual(scores.shape, (6,))
