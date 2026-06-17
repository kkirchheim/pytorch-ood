import unittest

import torch
from torch.utils.data import DataLoader

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import KNN
from tests.helpers import ClassificationModel, sample_dataset


class TestKNN(unittest.TestCase):
    """
    Tests for k-Nearest Neighbor
    """

    @unittest.skip("fails for currently unknown reasons")
    def test_requires_fitting(self):
        """
        TODO
        """
        model = ClassificationModel()
        detector = KNN(model)

        x = torch.zeros(size=(128, 10))

        with self.assertRaises(RequiresFittingException):
            detector(x)

        self.assertTrue(True)

    def test_input(self):
        """ """
        model = ClassificationModel()
        detector = KNN(model)

        ds = sample_dataset(n_dim=10)
        loader = DataLoader(ds)
        detector.fit(loader)

        x = torch.randn(size=(128, 10))
        scores = detector(x)

        print(scores)
        self.assertIsNotNone(scores)

    def test_k_neighbors(self):
        """The k-th nearest neighbor distance is non-decreasing in k."""
        g = torch.Generator().manual_seed(1)
        z_train = torch.randn(size=(200, 10), generator=g)
        y_train = torch.zeros(200, dtype=torch.long)
        z_test = torch.randn(size=(64, 10), generator=g)

        d1 = KNN(None, k=1).fit_features(z_train, y_train)
        d5 = KNN(None, k=5).fit_features(z_train, y_train)

        s1 = d1.predict_features(z_test)
        s5 = d5.predict_features(z_test)

        self.assertEqual(s1.shape, (64,))
        self.assertEqual(s5.shape, (64,))
        # distance to the 5th neighbor is at least the distance to the 1st
        self.assertTrue(torch.all(s5 >= s1 - 1e-6))

    def test_knn_kwargs_no_longer_conflict(self):
        """Passing sklearn kwargs must not collide with a hardcoded n_neighbors."""
        model = ClassificationModel()
        detector = KNN(model, k=3, metric="euclidean")
        ds = sample_dataset(n_dim=10, seed=3)
        detector.fit(DataLoader(ds))
        scores = detector(torch.randn(size=(16, 10)))
        self.assertEqual(scores.shape, (16,))
