import unittest

import torch

from src.pytorch_ood.loss import LogitNorm

torch.manual_seed(123)


class TestLogitNorm(unittest.TestCase):
    """
    Test code for energy bounded learning
    """

    def test_forward(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)

    def test_forward_only_positive(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)

    def test_forward_only_negative(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(128, 10))
        target = torch.ones(size=(128,)).long() * -1
        loss = criterion(logits, target)
        self.assertIsNotNone(loss)

    def test_set_alpha(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)

    def test_set_ms(self):
        criterion = LogitNorm()
        logits = torch.randn(size=(10, 10))
        target = torch.zeros(size=(10,)).long()
        target[5:] = -1

        loss = criterion(logits, target)

        self.assertIsNotNone(loss)

    def test_loss_is_non_negative(self):
        """
        Cross-entropy is always non-negative. The previous implementation applied
        nll_loss to normalized logits (instead of log-probabilities), which
        produced negative losses for confident predictions.
        """
        criterion = LogitNorm()
        logits = torch.tensor([[10.0, -5.0, -5.0]])
        target = torch.tensor([0])

        loss = criterion(logits, target)

        self.assertGreaterEqual(loss.item(), 0.0)

    def test_matches_reference_formula(self):
        """
        Loss must equal cross-entropy on temperature-scaled, L2-normalized logits,
        as defined in the LogitNorm paper.
        """
        t = 0.04
        criterion = LogitNorm(t=t)
        logits = torch.randn(size=(32, 10))
        target = torch.randint(0, 10, size=(32,))

        loss = criterion(logits, target)

        norm = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
        expected = torch.nn.functional.cross_entropy(logits / (t * norm), target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_scale_invariance(self):
        """
        Scaling the logits by a constant factor must not change the loss;
        this is the core property of LogitNorm.
        """
        criterion = LogitNorm(t=1.0)
        logits = torch.randn(size=(16, 10))
        target = torch.randint(0, 10, size=(16,))

        loss_1 = criterion(logits, target)
        loss_2 = criterion(100.0 * logits, target)

        self.assertTrue(torch.allclose(loss_1, loss_2, atol=1e-5))
