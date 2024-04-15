import unittest

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

from pytorch_ood.loss import CrossEntropyLoss
from pytorch_ood.utils import OODMetrics
from src.pytorch_ood.detector import MaxSoftmax, ViM
from tests.helpers import ClassificationModel, sample_dataset


class TestViM(unittest.TestCase):
    """
    Tests for ViM
    """

    def test_classification_input(self):
        model = ClassificationModel(n_hidden=10)
        w = model.classifier.weight.data
        b = model.classifier.bias.data
        detector = ViM(model.features, 2, w=w, b=b)

        y = torch.cat([torch.zeros(size=(10,)), torch.ones(size=(10,))])
        x = torch.randn(size=(20, 10))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.fit(loader)

        x = torch.zeros(size=(128, 10))
        y = detector(x)
        print(y)

        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))

    def test_logits(self):
        """
        Make sure the logit calculation in the detector is correct
        """
        torch.manual_seed(1234)
        n_dim = 20
        train = sample_dataset(centers=3, n_dim=n_dim, seed=123, n_samples=300)
        train_loader = DataLoader(train, batch_size=64, shuffle=True)

        model = ClassificationModel(num_inputs=n_dim, n_hidden=20)
        opti = SGD(model.parameters(), lr=0.01)

        criterion = CrossEntropyLoss()
        for epoch in range(2):
            for x, y in train_loader:
                opti.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                opti.step()

        x = torch.randn(size=(128, n_dim)) + torch.Tensor(n_dim * [0])

        model.eval()

        detector = ViM(
            model.features,
            d=5,
            w=model.classifier.weight.data,
            b=model.classifier.bias.data,
        )

        with torch.no_grad():
            logits1 = model(x)
            features = model.features(x)
        logits2 = detector._get_logits(features)
        for logit1, logit2 in zip(logits1.view(-1), logits2.view(-1)):
            self.assertAlmostEqual(logit1.item(), logit2.item(), places=5)

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

        x = torch.randn(size=(128, n_dim)) + torch.Tensor(n_dim * [0])
        y = torch.ones(size=(128,)) * -1

        detector = ViM(
            model.features,
            d=5,
            w=model.classifier.weight.data,
            b=model.classifier.bias.data,
        )
        detector.fit(train_loader)

        metrics = OODMetrics()
        metrics.update(detector(x), y)

        for x, y in test_loader:
            metrics.update(detector(x), y)
        print(metrics.compute())

        detector = MaxSoftmax(model)
        metrics = OODMetrics()

        x = torch.randn(size=(128, n_dim)) + torch.Tensor(n_dim * [0])
        y = torch.ones(size=(128,)) * -1
        metrics.update(detector(x), y)
        for x, y in test_loader:
            metrics.update(detector(x), y)
        metrics.update(detector(x), y)
        print(metrics.compute())
