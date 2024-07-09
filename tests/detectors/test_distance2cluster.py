import unittest
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.pytorch_ood.detector import Distance2Clusters  # Replace 'your_module' with the actual module name

class TestDistance2Clusters(unittest.TestCase):
    def setUp(self):
        # Create a simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, 2)
        )
        
        # Create some dummy data
        self.x = torch.randn(100, 10)
        self.y = torch.randint(0, 2, (100,))
        self.dataset = TensorDataset(self.x, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=32)

    def test_init(self):
        detector = Distance2Clusters(self.model)
        self.assertIsNone(detector.clusters)
        self.assertEqual(detector.subclusters, 1)

        clusters = torch.randn(7, 20)
        detector = Distance2Clusters(self.model, clusters=clusters, subclusters=10)
        self.assertIsNotNone(detector.clusters)
        self.assertEqual(detector.subclusters, 10)
        self.assertEqual(detector.n_classes, 2)



    def test_predict_features_without_y(self):
        clusters = torch.randn(7, 20)
        detector = Distance2Clusters(self.model,clusters=clusters, subclusters=10)

        # Generate some test features
        test_features = torch.randn(10, 7)
        
        distances,_ = detector.predict_features(test_features)
        
        self.assertEqual(distances.shape, (10,))
        self.assertTrue(torch.all(distances >= 0) and torch.all(distances <= 180))

    def test_predict_features_with_y(self):
        clusters = torch.randn(7, 20)

        detector = Distance2Clusters(self.model,clusters, subclusters=10)
        # Generate some test features and labels
        test_features = torch.randn(10, 7)
        test_labels = torch.randint(0, 2, (10,))
        
        distances, indices = detector.predict_features(test_features, test_labels)
        
        self.assertEqual(distances.shape, (10,))
        self.assertEqual(indices.shape, (10,))
        self.assertTrue(torch.all(distances >= 0))

if __name__ == '__main__':
    unittest.main()