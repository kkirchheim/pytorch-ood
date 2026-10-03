import unittest
import warnings

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

from pytorch_ood.loss import CrossEntropyLoss
from pytorch_ood.metrics import OODMetrics
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
        y = torch.arange(128) % 10

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

    @staticmethod
    def _paper_dists(logits):
        """d_k: mean posterior of the samples whose argmax is k (paper, Section 4)."""
        p = logits.softmax(dim=1)
        predictions = p.argmax(dim=1)
        return {int(k): p[predictions == k].mean(dim=0) for k in predictions.unique()}

    @staticmethod
    def _paper_scores(logits, dists):
        """min_k KL[p(y | x) || d_k] (paper, Section 4)."""
        p = logits.softmax(dim=1)
        kl = [(p * (p / d).log()).sum(dim=1) for d in dists.values()]
        return torch.stack(kl, dim=1).min(dim=1).values

    @staticmethod
    def _misclassified_logits(n=300, n_classes=3, seed=0):
        """Logits where 30% of the samples are predicted as another class than their label."""
        g = torch.Generator().manual_seed(seed)
        labels = torch.arange(n) % n_classes
        predicted = labels.clone()
        wrong = torch.rand(n, generator=g) < 0.3
        predicted[wrong] = (labels[wrong] + 1) % n_classes
        logits = torch.randn(n, n_classes, generator=g) + 3 * torch.eye(n_classes)[predicted]
        return logits, labels

    def test_fit_groups_by_predicted_class(self):
        logits, labels = self._misclassified_logits()
        detector = KLMatching(None).fit_logits(logits, labels)
        expected = self._paper_dists(logits)
        self.assertEqual(sorted(detector.dists.keys()), ["0", "1", "2"])
        for k, d_k in expected.items():
            torch.testing.assert_close(detector.dists[str(k)].data, d_k)

    def test_fit_does_not_need_labels(self):
        logits, labels = self._misclassified_logits()
        with_labels = KLMatching(None).fit_logits(logits, labels)
        without_labels = KLMatching(None).fit_logits(logits)
        shuffled = KLMatching(None).fit_logits(logits, labels[torch.randperm(len(labels))])
        test = torch.randn(20, 3)
        expected = with_labels.predict_logits(test)
        torch.testing.assert_close(without_labels.predict_logits(test), expected)
        torch.testing.assert_close(shuffled.predict_logits(test), expected)

    def test_score_is_min_kl_over_classes(self):
        logits, labels = self._misclassified_logits(n=3000, n_classes=10)
        detector = KLMatching(None).fit_logits(logits, labels)
        # flat posteriors, for which the closest typical posterior often is not the one of
        # the predicted class
        test = torch.randn(500, 10, generator=torch.Generator().manual_seed(1))
        p = test.softmax(dim=1)
        dists = self._paper_dists(logits)
        kl = torch.stack([(p * (p / d).log()).sum(dim=1) for d in dists.values()], dim=1)
        self.assertTrue((kl.argmin(dim=1) != p.argmax(dim=1)).any())
        torch.testing.assert_close(detector.predict_logits(test), self._paper_scores(test, dists))

    def test_classes_never_predicted_are_not_needed(self):
        """Classes without a typical posterior are left out of the minimum."""
        logits, labels = self._misclassified_logits()
        logits[:, 2] -= 20  # class 2 is never predicted
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector = KLMatching(None).fit_logits(logits, labels)
        self.assertEqual(sorted(detector.dists.keys()), ["0", "1"])
        test = torch.randn(20, 3)
        test[:, 2] += 5  # predicted as class 2
        scores = detector.predict_logits(test)
        torch.testing.assert_close(scores, self._paper_scores(test, self._paper_dists(logits)))

    def test_ignores_ood_samples(self):
        """OOD samples neither get a distribution nor change the ones of known classes."""
        logits, labels = self._misclassified_logits()
        ood_logits = torch.randn(50, 3) * 5
        detector = KLMatching(None).fit_logits(
            torch.cat([logits, ood_logits]),
            torch.cat([labels, -torch.ones(50, dtype=torch.long)]),
        )
        reference = KLMatching(None).fit_logits(logits, labels)
        self.assertEqual(sorted(detector.dists.keys()), sorted(reference.dists.keys()))
        for k in reference.dists.keys():
            torch.testing.assert_close(detector.dists[k], reference.dists[k])

    def test_predict_logits_before_fit_raises(self):
        with self.assertRaises(RequiresFittingException):
            KLMatching(None).predict_logits(torch.randn(4, 3))

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
