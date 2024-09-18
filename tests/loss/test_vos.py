import unittest

import torch

from src.pytorch_ood.loss import VirtualOutlierSynthesizingRegLoss, VOSRegLoss
from tests.helpers.model import ClassificationModel, SegmentationModel

torch.manual_seed(123)


class TestVOSRegularization(unittest.TestCase):
    """
    Test code for VOS regularization loss
    """

    def init_loss(self, num_classes, reduction="mean", alpha=0.1):
        weights_energy = torch.nn.Linear(num_classes, 1).cpu()
        torch.nn.init.uniform_(weights_energy.weight)
        phi = torch.nn.Linear(1, 2).cpu()
        criterion = VOSRegLoss(phi, weights_energy, alpha=alpha, device="cpu", reduction=reduction)
        return criterion

    def test_forward(self):
        criterion = self.init_loss(10)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_positive(self):
        criterion = self.init_loss(10)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_negative(self):
        criterion = self.init_loss(10)
        logits = torch.randn(size=(128, 10))
        target = torch.ones(size=(128,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_set_alpha(self):
        criterion = self.init_loss(10, alpha=2)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_segmentation(self):
        model = SegmentationModel()
        criterion = self.init_loss(3, reduction="sum")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        # self.assertEqual(loss.shape, (10, 32, 32))
        # None Loss geht nicht , weil unterschiedliche größe von loss und uncertain loss
        self.assertNotEqual(loss, 0)
        loss.mean().backward()


class TestVirtualOutlierSynthesizingRegLoss(unittest.TestCase):
    """
    Test code for VirtualOutlierSynthesizingRegLoss
    """

    def init_loss(self, num_classes, reduction="mean", alpha=0.1):
        weights_energy = torch.nn.Linear(num_classes, 1).cpu()
        torch.nn.init.uniform_(weights_energy.weight)
        phi = torch.nn.Linear(1, 2).cpu()
        model = ClassificationModel()
        criterion = VirtualOutlierSynthesizingRegLoss(
            phi,
            weights_energy,
            alpha=alpha,
            device="cpu",
            reduction=reduction,
            num_classes=num_classes,
            num_input_last_layer=10,
            fc=model.classifier,
            sample_number=5,
            sample_from=8,
        )
        return criterion, model

    def test_forward_only_positive(self):
        criterion, model = self.init_loss(10)
        x = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()

        features = model.features(x)
        logits = model.classifier(features)
        loss = criterion(logits, features, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_negative(self):
        criterion, model = self.init_loss(10)
        x = torch.randn(size=(128, 10))

        target = torch.ones(size=(128,)).long() * -1
        features = model.features(x)
        logits = model.classifier(features)

        with self.assertRaises(ValueError) as context:
            loss = criterion(logits, features, target)
        self.assertEqual(
            str(context.exception),
            "Outlier targets in VirtualOutlierSynthesizingRegLoss. This loss function only supports inlier targets.",
        )

    def test_forward_set_alpha(self):
        criterion, model = self.init_loss(10, alpha=2)
        x = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()

        features = model.features(x)
        logits = model.classifier(features)
        loss = criterion(logits, features, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
