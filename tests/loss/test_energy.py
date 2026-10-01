import unittest

import torch

from src.pytorch_ood.loss import EnergyRegularizedLoss
from tests.helpers.model import SegmentationModel

torch.manual_seed(123)


class TestEnergyRegularization(unittest.TestCase):
    """
    Test code for energy bounded learning
    """

    def test_forward(self):
        criterion = EnergyRegularizedLoss()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_positive(self):
        criterion = EnergyRegularizedLoss()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_forward_only_negative(self):
        criterion = EnergyRegularizedLoss(margin_out=1.0)
        logits = torch.randn(size=(128, 10))
        target = torch.ones(size=(128,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_set_alpha(self):
        criterion = EnergyRegularizedLoss(alpha=2)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_set_ms(self):
        criterion = EnergyRegularizedLoss(margin_in=-4, margin_out=-8)
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_segmentation_with_unknown(self):
        model = SegmentationModel()
        # the OOD hinge max(0, m_out - E) is active while the energy is below m_out
        criterion = EnergyRegularizedLoss(margin_out=5.0, reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertNotEqual(loss[0, 0, 0], 0)
        loss.mean().backward()

    def test_segmentation_with_unknown_set_similar_ms(self):
        model = SegmentationModel()
        criterion = EnergyRegularizedLoss(margin_in=5, margin_out=5, reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertNotEqual(loss[0, 0, 0], 0)
        loss.mean().backward()

    def test_segmentation_with_unknown_set_different_ms(self):
        model = SegmentationModel()
        criterion = EnergyRegularizedLoss(margin_in=-5, margin_out=5, reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()
        target[0, 0, 0] = -1

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertNotEqual(loss[0, 0, 0], 0)
        loss.mean().backward()

    def test_segmentation_forward_only_positive(self):
        model = SegmentationModel()
        criterion = EnergyRegularizedLoss(reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long()

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.mean(), 0)

    def test_segmentation_forward_only_negative(self):
        model = SegmentationModel()
        criterion = EnergyRegularizedLoss(reduction="none")
        x = torch.randn(size=(10, 3, 32, 32))
        target = torch.zeros(size=(10, 32, 32)).long() * -1

        logits = model(x)
        loss = criterion(logits, target)
        self.assertEqual(loss.shape, (10, 32, 32))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.mean(), 0)

    def test_matches_paper_formula(self):
        """CE over ID samples plus separately averaged ID and OOD hinges (Liu et al., 2020)."""
        criterion = EnergyRegularizedLoss(alpha=0.5, margin_in=-3.0, margin_out=1.0)
        logits = torch.randn(size=(12, 4))
        target = torch.arange(12) % 5 - 1  # some -1
        known = target >= 0
        energy = -torch.logsumexp(logits, dim=1)
        expected = torch.nn.functional.cross_entropy(logits[known], target[known]) + 0.5 * (
            (energy[known] + 3.0).relu().pow(2).mean()
            + (1.0 - energy[~known]).relu().pow(2).mean()
        )
        torch.testing.assert_close(criterion(logits, target), expected)

    def test_segmentation_equals_pixels_as_samples(self):
        criterion = EnergyRegularizedLoss(margin_in=-3.0, margin_out=1.0)
        logits = torch.randn(size=(2, 4, 5, 5))
        target = torch.arange(50).reshape(2, 5, 5) % 5 - 1
        flat = criterion(logits.permute(0, 2, 3, 1).reshape(-1, 4), target.reshape(-1))
        torch.testing.assert_close(criterion(logits, target), flat)
