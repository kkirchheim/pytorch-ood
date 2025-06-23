import unittest

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.functional import F
from tests.helpers import ClassificationModel, sample_dataset

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.detector import NCI
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics


class TestASH(unittest.TestCase):
    """
    Tests for activation shaping
    """

    def test_nofitting(self):
        """ """
        model = WideResNet(num_classes=10).eval()

        with self.assertRaises(RequiresFittingException):
            detector = NCI(
                encoder=model.features_before_pool,
                head=model.forward_from_before_pool,
            )

            x = torch.randn(size=(16, 3, 32, 32))

            detector(x)

    def test_full(self):
        """ """

        torch.manual_seed(42)
        dataset = sample_dataset(n_samples=1000, n_dim=10, centers=3, seed=42, loc=3)
        loader = DataLoader(dataset, batch_size=1000)

        model = self._train_model(loader)

        detector = NCI(encoder=model.layer1, head=model.classifier, alpha=0.0)

        detector.fit(loader)

        x, y = next(iter(loader))
        outputs = detector.predict(x)

        dataset = sample_dataset(n_samples=100, n_dim=10, centers=3, std=5, seed=42, loc=-3)
        loader = DataLoader(dataset, batch_size=1000)
        x, _ = next(iter(loader))
        outputs2 = detector.predict(x)

        m = OODMetrics()
        m.update(outputs, y)
        m.update(outputs2, -torch.ones_like(outputs2))

        results = m.compute()

        print(results["AUROC"])
        self.assertGreater(results["AUROC"], 0.9)

    def _train_model(self, loader):
        model = ClassificationModel(num_outputs=3, n_hidden=128).eval()
        sgd = SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9, nesterov=True)
        for i in range(20):
            for x, y in loader:
                sgd.zero_grad()
                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                sgd.step()
                print(f"Accuracy: {(y_hat.argmax(dim=1) == y).float().mean()}")
        return model
