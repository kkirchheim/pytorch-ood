import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import Mahalanobis
from tests.helpers import ClassificationModel


class MahalanobisTest(unittest.TestCase):
    """
    Test mahalanobis method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_something(self):
        nn = ClassificationModel()
        model = Mahalanobis(nn)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(scores)

        scores = model(torch.ones(size=(10, 10)) * 10)
        print(scores)

        self.assertIsNotNone(scores)

    def test_nofit(self):
        nn = ClassificationModel()
        model = Mahalanobis(nn)
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            model(x)
            
    def test_mu_shape(self):
        number_classes = 5
        embedding_size = 10
        nn = ClassificationModel(num_inputs=30, num_outputs=number_classes, embedding_dim=embedding_size)

        x = torch.randn(size=(64,30))
        y = torch.randint(0, number_classes, (64,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset,batch_size=64)

        model = Mahalanobis(nn.encoder)
        model.fit(loader)
        
        self.assertEqual(model.mu.shape[0], number_classes)
        self.assertEqual(model.mu.shape[1], embedding_size)
        
