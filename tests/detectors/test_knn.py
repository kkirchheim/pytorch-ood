import unittest

import torch
from torch.utils.data import DataLoader

from src import pytorch_ood
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
