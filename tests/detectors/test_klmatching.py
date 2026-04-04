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

    def test_predict_logits(self):
        """predict_logits works directly on logits."""
        torch.manual_seed(0)
        n_classes = 3
        model = ClassificationModel(num_outputs=n_classes)
        detector = KLMatching(model)

        # Fit on ID data (class labels 0, 1, 2)
        x_fit = torch.randn(90, 10)
        y_fit = torch.repeat_interleave(torch.arange(n_classes), 30)
        detector.fit(DataLoader(TensorDataset(x_fit, y_fit), batch_size=30))

        logits = torch.randn(16, n_classes)
        scores = detector.predict_logits(logits)
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_mock_performance(self):
        """
        Train a model on well-separated Gaussians and verify AUROC > 0.95.
        KL-Matching estimates typical posteriors per class and scores OOD samples
        by KL divergence from those typical posteriors.
        """
        torch.manual_seed(7)
        n_dim, n_classes, n_per_class = 20, 3, 200

        centers = torch.eye(n_classes, n_dim) * 8.0
        g = torch.Generator().manual_seed(7)

        x_train = torch.cat(
            [
                torch.randn(n_per_class, n_dim, generator=g) * 0.3 + centers[c]
                for c in range(n_classes)
            ]
        )
        y_train = torch.repeat_interleave(torch.arange(n_classes), n_per_class)

        model = ClassificationModel(num_inputs=n_dim, num_outputs=n_classes, n_hidden=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(x_train), y_train).backward()
            optimizer.step()
        model.eval()

        val_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64)
        detector = KLMatching(model)
        detector.fit(val_loader)

        # ID test set: same distribution
        x_id = torch.cat(
            [torch.randn(50, n_dim, generator=g) * 0.3 + centers[c] for c in range(n_classes)]
        )
        y_id = torch.repeat_interleave(torch.arange(n_classes), 50)

        # OOD test set: far-away cluster
        x_ood = torch.randn(150, n_dim, generator=g) * 0.3 + 30.0
        y_ood = torch.full((150,), -1, dtype=torch.long)

        metrics = OODMetrics()
        with torch.no_grad():
            metrics.update(detector(x_id), y_id)
            metrics.update(detector(x_ood), y_ood)

        results = metrics.compute()
        self.assertGreater(
            results["AUROC"], 0.95, f"Expected AUROC > 0.95, got {results['AUROC']:.4f}"
        )

    @unittest.skip(reason="Requires GPU")
    def test_gpu(self):
        device = "cuda:0"
        model = ClassificationModel().to(device)
        detector = KLMatching(model)

        x = torch.zeros(size=(128, 10))
        y = torch.randint(3, size=(128,))

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        detector.to(device)
        detector.fit(loader)
        with torch.no_grad():
            y = detector(x.to(device))

        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (128,))
