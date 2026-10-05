import unittest

import torch

from src.pytorch_ood.model import GRUClassifier


class TestGRUClassifier(unittest.TestCase):
    def test_embedding_dim(self):
        for embedding_dim in [32, 50, 64]:
            model = GRUClassifier(num_classes=3, n_vocab=10, embedding_dim=embedding_dim)
            x = torch.randint(0, 10, (2, 5))
            self.assertEqual(model(x).shape, (2, 3))
            self.assertEqual(model.features(x).shape, (2, 128))
