import unittest

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

from pytorch_ood.loss import CrossEntropyLoss
from pytorch_ood.utils import OODMetrics
from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector.klmatching import KLMatching
from tests.helpers import ClassificationModel, sample_dataset


class TestKLMatching(unittest.TestCase):
    """
    Tests for KL Matching
    """

    def test_classification_input(self):
        model = ClassificationModel()
        detector = KLMatching(model)

        x = torch.zeros(size=(128, 10))
        y = torch.arange(128) % 5

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader)
        self.assertGreater(len(detector.dists), 0)

        with torch.no_grad():
            y = detector(x)

        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))

    def test_no_fit(self):
        model = ClassificationModel()
        detector = KLMatching(model)
        x = torch.zeros(size=(128, 10))

        with self.assertRaises(RequiresFittingException):
            detector(x)

    def test_train(self):
        torch.manual_seed(1234)

        n_dim = 20
        lengths = [300, 300, 300]

        ds = sample_dataset(centers=3, n_dim=n_dim, seed=123, n_samples=300)
        train, val, test = random_split(ds, lengths=lengths)

        train_loader = DataLoader(train, batch_size=64, shuffle=True)
        val_loader = DataLoader(val, batch_size=64, shuffle=True)
        test_loader = DataLoader(test, batch_size=64, shuffle=True)

        model = ClassificationModel(num_inputs=n_dim, n_hidden=20)
        opti = SGD(model.parameters(), lr=0.01)

        criterion = CrossEntropyLoss()
        for epoch in range(10):
            for x, y in train_loader:
                opti.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                print(loss.item())
                loss.backward()
                opti.step()

        model.eval()
        detector = KLMatching(model)
        detector.fit(val_loader)

        metrics = OODMetrics()
        for x, y in test_loader:
            metrics.update(detector(x), y)

        # create ood samples
        x = torch.randn(size=(128, n_dim)) + torch.Tensor(n_dim * [0])
        y = torch.ones(size=(128,)) * -1
        metrics.update(detector(x), y)

        self.assertGreater(metrics.compute()["AUROC"], 0.99)

    @unittest.skip(reason="Requires GPU")
    def test_gpu(self):
        device = "cuda:0"
        model = ClassificationModel().to(device)
        detector = KLMatching(model)

        x = torch.zeros(size=(128, 10))
        y = torch.randint(3, size=(128,))

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader, device=device)
        with torch.no_grad():
            y = detector(x.to(device))

        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))
