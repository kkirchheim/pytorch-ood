import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from pytorch_ood.detector import Distance2Clusters
from pytorch_ood.api import RequiresFittingException
import matplotlib.pyplot as plt
class NormalizedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        
    def forward(self, x):
        return F.normalize(super().forward(x), dim=1)

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_clusters_per_class):
        super().__init__()
        self.input_dim = input_dim
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim,20),
            nn.ReLU(),
            nn.Linear(20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes * num_clusters_per_class)
        self.num_classes = num_classes
        self.num_clusters_per_class = num_clusters_per_class
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        # Reshape the output to (batch_size, num_classes, num_clusters_per_class)
        x = x.view(-1, self.num_classes, self.num_clusters_per_class)
        # Sum over the clusters for each class
        x = x.sum(dim=2)
        return x

class TestDistance2Clusters(unittest.TestCase):
    def setUp(self):
        self.input_dim = 2
        self.hidden_dim = 2
        self.num_classes = 2
        self.num_clusters_per_class = 2
        self.model = SimpleModel(self.input_dim, self.hidden_dim, self.num_classes, self.num_clusters_per_class)
        # generate data centered around [1, 1],  [1,-1] for class 1 
        # and for class 2 center around [-1,1] and [-1,-1]  
        self.X = torch.cat([torch.randn(1000, 2)/10 + torch.tensor([1, 1]),
                            torch.randn(1000, 2)/10 + torch.tensor([1, -1]),
                            torch.randn(1000, 2)/10 + torch.tensor([-1, 1]),
                            torch.randn(1000, 2)/10 + torch.tensor([-1, -1])])
        
        
        self.y = torch.cat([torch.zeros(2000), torch.ones(2000)]).long()
        self.dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        # train the model
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(10):
            for x, y in self.dataloader:
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
        # get the cluster centers, which are the weights of the classifier layer 
    
        self.cluster_centers = self.model.classifier.weight.data.T

        
    def test_init(self):
        detector = Distance2Clusters(self.model.feature_extractor, cluster_centers=self.cluster_centers, 
                                     subclusters=self.num_clusters_per_class, n_classes=self.num_classes)
        self.assertIsNotNone(detector.cluster_centers)
        self.assertEqual(detector.subclusters, self.num_clusters_per_class)
        self.assertEqual(detector.n_classes, self.num_classes)
        # evaluate the model
        y_pred=self.model(self.X)
        # print the accuracy
        print('##########MODEL ACCURACY##########')
        print((y_pred.argmax(dim=1)==self.y).float().mean())
        

    def test_init_invalid_clusters(self):
        invalid_clusters = F.normalize(torch.randn(self.hidden_dim, self.num_classes * (self.num_clusters_per_class + 1)), dim=0)
        with self.assertRaises(AssertionError):
            Distance2Clusters(self.model.feature_extractor, cluster_centers=invalid_clusters, 
                              subclusters=self.num_clusters_per_class, n_classes=self.num_classes)
    def test_in_distribution_performance(self):
        detector = Distance2Clusters(self.model, cluster_centers=self.cluster_centers, 
                                     subclusters=self.num_clusters_per_class, n_classes=self.num_classes)
        in_dist_samples = self.X[:100]  # Use a subset of the training data
        distances = detector.predict(in_dist_samples)
        self.assertTrue(torch.mean(distances) < 90), 


    def test_center_to_center(self):
        detector = Distance2Clusters(self.model, cluster_centers=self.cluster_centers, 
                                     subclusters=self.num_clusters_per_class, n_classes=self.num_classes)
        x_test = torch.tensor([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=torch.float32)
        distances = detector.predict(x_test)
        self.assertTrue(torch.mean(distances) < 60,
                        f"In-distribution samples should have low distance to the cluster centers got{ torch.mean(distances)}")



    def test_model_performance(self):
        test_samples = self.X[-100:]  # Use last 100 samples as test set
        test_labels = self.y[-100:]
        predictions = self.model(test_samples).argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean()
        self.assertGreater(accuracy, 0.8)  # Expect at least 80% accuracy


if __name__ == '__main__':
    unittest.main()