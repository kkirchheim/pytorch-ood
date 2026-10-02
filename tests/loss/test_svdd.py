import unittest

import torch
from torch.optim import SGD

from src.pytorch_ood.loss import DeepSADLoss, DeepSVDDLoss
from tests.helpers import ClassificationModel


class TestDeepSVDD(unittest.TestCase):
    def test_forward(self):
        criterion = DeepSVDDLoss(n_dim=10, reduction=None)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertTrue((loss[5:] == 0).all())

    def test_radius(self):
        model = ClassificationModel(n_hidden=1024)
        opti = SGD(model.parameters(), lr=0.01)
        criterion = DeepSVDDLoss(n_dim=3, radius=1.0, reduction=None)

        x = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()

        target[5:] = -1

        for i in range(1000):
            loss = criterion(model(x), target)

            loss.mean().backward()
            opti.zero_grad()
            opti.step()

        print(loss)
        print(criterion.distance(model(x)))

        self.assertIsNotNone(loss)
        self.assertTrue((loss[5:] == 0).all())


class TestDeepSAD(unittest.TestCase):
    def test_forward(self):
        criterion = DeepSADLoss(n_dim=10)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)

    def test_forward_2(self):
        criterion = DeepSADLoss(n_dim=10, reduction=None)
        logits = torch.randn(size=(10, 10))
        target = -1 * torch.ones(size=(10,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertFalse((loss == 0).any())

    def test_deprecated_name(self):
        import src.pytorch_ood.loss as losses

        with self.assertWarns(DeprecationWarning):
            self.assertIs(losses.SSDeepSVDDLoss, DeepSADLoss)


class TestDeepSADValues(unittest.TestCase):
    def _loss(self, **kwargs):
        criterion = DeepSADLoss(n_dim=2, reduction="none", **kwargs)
        criterion.center.params.data.zero_()
        return criterion

    def test_squared_distance(self):
        x = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        loss = self._loss()(x, torch.tensor([0, -1]))
        torch.testing.assert_close(loss, torch.tensor([4.0, 1 / (4.0 + 1e-6)]))

    def test_eta_weights_outliers(self):
        x = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        loss = self._loss(eta=3.0)(x, torch.tensor([0, -1]))
        torch.testing.assert_close(loss, torch.tensor([4.0, 3 / (4.0 + 1e-6)]))

    def test_outlier_at_center_is_finite(self):
        loss = self._loss()(torch.zeros(1, 2), torch.tensor([-1]))
        self.assertTrue(torch.isfinite(loss).all())


class TestSVDDLossDefaultRadius(unittest.TestCase):
    def test_default_radius(self):
        criterion = DeepSVDDLoss(n_dim=2)
        loss = DeepSVDDLoss.svdd_loss(torch.randn(3, 2), criterion.center)
        self.assertEqual(loss.shape, (3,))


class TestDeepSADCenterAndDistance(unittest.TestCase):
    def test_center_argument(self):
        center = torch.tensor([1.0, 2.0])
        criterion = DeepSADLoss(n_dim=2, center=center)
        torch.testing.assert_close(criterion.center.params, center.reshape(1, 2))

    def test_distance_is_squared_distance_to_center(self):
        criterion = DeepSADLoss(n_dim=2, center=torch.tensor([1.0, 0.0]))
        x = torch.tensor([[1.0, 0.0], [3.0, 0.0], [1.0, -1.0]])
        torch.testing.assert_close(criterion.distance(x), torch.tensor([0.0, 4.0, 1.0]))
