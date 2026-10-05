"""
Unsupervised losses discard OOD samples (targets < 0): the loss on a batch with outliers equals the
loss on the same batch without them, and a batch of outliers only gives a zero loss with gradients.
"""

import copy
import unittest

import torch
import torch.nn.functional as F

from pytorch_ood import loss
from pytorch_ood.api import Task
from tests.test_info import LOSS_INPUTS, LOSSES, SHAPES

UNSUPERVISED = [cls for cls in LOSSES if not cls.info.supervised]


def _batch(cls):
    criterion, inputs = LOSS_INPUTS[cls](SHAPES[Task.CLASSIFICATION])
    # the target is the last argument
    *tensors, target = inputs
    return criterion, tensors, target


class TestUnsupervisedLossesDiscardOOD(unittest.TestCase):
    def test_covers_all_unsupervised_losses(self):
        self.assertEqual(len(UNSUPERVISED), 8)

    def test_outliers_do_not_change_the_loss(self):
        for cls in UNSUPERVISED:
            with self.subTest(loss=cls.__name__):
                torch.manual_seed(0)
                criterion, tensors, target = _batch(cls)
                target = target.clone()
                target[::3] = -1
                known = target >= 0
                reference = copy.deepcopy(criterion)

                with self.assertWarns(UserWarning):
                    mixed = criterion(*tensors, target)
                without = reference(*(t[known] for t in tensors), target[known])

                torch.testing.assert_close(mixed, without)

    def test_only_outliers(self):
        for cls in UNSUPERVISED:
            with self.subTest(loss=cls.__name__):
                criterion, tensors, target = _batch(cls)
                tensors = [t.clone().requires_grad_(True) for t in tensors]

                with self.assertWarns(UserWarning):
                    value = criterion(*tensors, torch.full_like(target, -1))

                self.assertEqual(value.shape, ())
                self.assertEqual(value.item(), 0)
                value.backward()
                for t in tensors:
                    self.assertTrue((t.grad == 0).all())

    def test_no_warning_without_outliers(self):
        for cls in UNSUPERVISED:
            with self.subTest(loss=cls.__name__):
                criterion, tensors, target = _batch(cls)
                with warnings_as_errors():
                    criterion(*tensors, target.clamp(min=0))


class warnings_as_errors:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", UserWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


class TestCrossEntropySegmentation(unittest.TestCase):
    def test_mean_over_id_pixels(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        target[0, :2] = -1
        known = target >= 0

        with self.assertWarns(UserWarning):
            value = loss.CrossEntropyLoss()(logits, target)

        expected = F.cross_entropy(logits.permute(0, 2, 3, 1)[known], target[known])
        torch.testing.assert_close(value, expected)

    def test_none_keeps_shape(self):
        target = torch.randint(0, 3, (2, 4, 4))
        target[0, 0] = -1
        with self.assertWarns(UserWarning):
            value = loss.CrossEntropyLoss(reduction="none")(torch.randn(2, 3, 4, 4), target)
        self.assertEqual(value.shape, (2, 4, 4))
        self.assertTrue((value[0, 0] == 0).all())


class TestDropUnknown(unittest.TestCase):
    def test_rejects_non_vector_targets(self):
        from pytorch_ood.utils import drop_unknown

        with self.assertRaises(ValueError):
            drop_unknown(torch.zeros(2, 4, 4).long(), torch.zeros(2, 3, 4, 4))


class TestSupervisedLossesDoNotWarn(unittest.TestCase):
    """Supervised losses that use unsupervised ones internally pass them only ID samples."""

    def test_mchad(self):
        criterion, inputs = LOSS_INPUTS[loss.MCHADLoss](SHAPES[Task.CLASSIFICATION])
        distmat, target = inputs
        target = target.clone()
        target[::3] = -1
        with warnings_as_errors():
            criterion(distmat, target)
