import copy
import unittest
from unittest.mock import patch

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
        criterion = VOSRegLoss(phi, weights_energy, alpha=alpha, reduction=reduction)
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
        model = ClassificationModel(num_outputs=num_classes)
        criterion = VirtualOutlierSynthesizingRegLoss(
            phi,
            weights_energy,
            alpha=alpha,
            reduction=reduction,
            fc=model.classifier,
            sample_number=5,
            sample_from=8,
        )
        return criterion, model

    def test_dimensions_from_fc(self):
        criterion, model = self.init_loss(num_classes=4)
        self.assertEqual(criterion.num_classes, 4)
        self.assertEqual(criterion.data_dict.shape, (4, 5, model.classifier.in_features))

    def test_queues_not_in_state_dict(self):
        # the fill levels are not saved either, so restoring only the queues would be inconsistent
        criterion, _ = self.init_loss(num_classes=3)
        self.assertNotIn("data_dict", criterion.state_dict())
        self.assertNotIn("eye_matrix", criterion.state_dict())

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
        # 3 classes, as output by the model (the energy of the outliers is computed now)
        criterion, model = self.init_loss(3)
        x = torch.randn(size=(128, 10))

        target = torch.ones(size=(128,)).long() * -1
        features = model.features(x)
        logits = model.classifier(features)

        # no ID samples: no cross-entropy and no energy regularization
        loss = criterion(logits, features, target)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(sum(criterion.number_dict.values()), 0)

    def test_real_outliers_are_ignored(self):
        criterion, model = self.init_loss(3)
        x = torch.randn(size=(12, 10))
        target = torch.tensor([0, 1, 2, -1] * 3)
        features = model.features(x)
        logits = model.classifier(features)

        reg = criterion._regularization_classification(logits, features, target)

        # queues are not full yet and real outliers are ignored: no regularization
        self.assertEqual(reg.item(), 0)
        # only the 9 ID samples were stored
        self.assertEqual(sum(criterion.number_dict.values()), 9)

        # changing the outliers does not change the loss
        unknown = target < 0
        changed_logits, changed_features = logits.clone(), features.clone()
        changed_logits[unknown] += 10
        changed_features[unknown] += 10
        first, _ = self.init_loss(3)
        second = copy.deepcopy(first)
        torch.testing.assert_close(
            first(logits, features, target), second(changed_logits, changed_features, target)
        )

    def test_full_queue_update_is_identical_without_cpu_transfer(self):
        criterion, model = self.init_loss(3)
        criterion.data_dict.copy_(
            torch.arange(criterion.data_dict.numel()).reshape_as(criterion.data_dict)
        )
        criterion.number_dict = {
            index: criterion.sample_number for index in range(criterion.num_classes)
        }
        reference = copy.deepcopy(criterion)

        features = torch.arange(70, dtype=torch.float32).reshape(7, 10)
        target = torch.tensor([2, 0, 2, 1, 2, 0, 2])
        prediction = model.classifier(features)

        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            reference.data_dict[dict_key] = torch.cat(
                (
                    reference.data_dict[dict_key][1:],
                    features[index].detach().view(1, -1),
                ),
                0,
            )

        torch.manual_seed(123)
        with patch.object(
            torch.Tensor,
            "cpu",
            side_effect=AssertionError("steady-state queue update transferred to CPU"),
        ):
            loss = criterion._regularization_classification(prediction, features, target)

        torch.manual_seed(123)
        reference_loss = reference._energy_regularization(
            prediction,
            torch.ones(len(target), dtype=torch.bool),
            reference._sample_virtual_outliers(),
        )

        self.assertTrue(torch.equal(criterion.data_dict, reference.data_dict))
        self.assertTrue(torch.equal(loss, reference_loss))

    def test_forward_set_alpha(self):
        criterion, model = self.init_loss(10, alpha=2)
        x = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()

        features = model.features(x)
        logits = model.classifier(features)
        loss = criterion(logits, features, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
