import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import fDBD
from tests.helpers import ClassificationModel


class fDBDTest(unittest.TestCase):
    """
    Test fDBD detector
    """

    def setUp(self) -> None:
        torch.manual_seed(123)
        self.nn = ClassificationModel()
        self.head = self.nn.classifier

    def test_fit_predict(self):
        # Wrap the features method as a module
        encoder = torch.nn.Sequential(self.nn.layer1, torch.nn.Tanh())
        detector = fDBD(encoder, self.head)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader)
        scores = detector(x)

        self.assertEqual(scores.shape, (20,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fit_predict_features(self):
        detector = fDBD(encoder=None, head=self.head)

        z = torch.randn(size=(20, 10))
        y = torch.zeros(size=(20,))

        detector.fit_features(z)
        scores = detector.predict_features(z)

        self.assertEqual(scores.shape, (20,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_nofit(self):
        detector = fDBD(self.nn.features, self.head)
        z = torch.randn(size=(5, 10))

        with self.assertRaises(RequiresFittingException):
            detector.predict_features(z)

    def test_ood_scores_higher(self):
        """ID points near class centers should score lower (less OOD) than far-away points."""
        detector = fDBD(self.nn.features, self.head)

        # create well-separated training data
        z_train = torch.cat(
            [
                torch.randn(50, 10) + 5,  # class 0 cluster
                torch.randn(50, 10) - 5,  # class 1 cluster
            ]
        )
        y_train = torch.cat([torch.zeros(50), torch.ones(50)])
        detector.fit_features(z_train)

        # ID-like features (near training clusters)
        z_id = torch.randn(20, 10) + 5
        # OOD-like features (far from everything)
        z_ood = torch.randn(20, 10) * 100

        scores_id = detector.predict_features(z_id)
        scores_ood = detector.predict_features(z_ood)

        # on average, OOD scores should be higher
        self.assertGreater(scores_ood.mean().item(), scores_id.mean().item())
