"""
Every loss must follow ``loss.to(device)`` like any :class:`torch.nn.Module`: its state moves
along, and with inputs on that device it computes the loss and its gradients there.

Losses should also work when they are not moved, as long as the inputs share a device. Losses
for which this is impossible are listed in ``REQUIRES_TO`` and document that ``.to()`` is required.
"""

import unittest
import warnings
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood import loss
from tests.test_info import LOSS_INPUTS, LOSSES, SHAPES

# the last GPU, so that code hard-coded to cuda:0 fails on machines with several GPUs
DEVICE = torch.device("cuda", max(torch.cuda.device_count() - 1, 0))
# stateful losses (VOS queues, running centers) only take some code paths after a few steps
STEPS = 3
REDUCTIONS = ("mean", "sum", "none")

REQUIRES_TO = {
    # the centers, which also compute the distances that forward() takes
    loss.CACLoss,
    loss.CenterLoss,
    loss.DeepSADLoss,
    loss.DeepSVDDLoss,
    loss.IILoss,
    loss.MCHADLoss,
    # logistic regression, energy weights and (for synthesis) the classifier and the queues
    loss.VOSRegLoss,
    loss.VirtualOutlierSynthesizingRegLoss,
}

# losses whose value depends on random draws, which differ between CPU and CUDA
RANDOM = {loss.VirtualOutlierSynthesizingRegLoss}


def _move(arg, device, all_ood: bool):
    if isinstance(arg, nn.Module):
        return arg.to(device)
    if arg.dtype == torch.long:  # targets
        return (torch.full_like(arg, -1) if all_ood else arg).to(device)
    return arg.to(device).requires_grad_()


def _tensors(name, value):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _tensors(f"{name}[{i}]", item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(f"{name}[{key!r}]", item)


def _tensor_state(module: nn.Module):
    """
    Tensors a module holds, including plain attributes and containers that ``to()`` does not
    move.
    """
    for name, sub in module.named_modules():
        prefix = f"{name}." if name else ""
        for key, value in vars(sub).items():
            if key not in ("_parameters", "_buffers", "_modules"):
                yield from _tensors(prefix + key, value)
        yield from ((prefix + k, t) for k, t in sub.named_parameters(recurse=False))
        yield from ((prefix + k, t) for k, t in sub.named_buffers(recurse=False))


def _cases(losses):
    for cls in losses:
        for task in sorted(cls.info.tasks):
            for all_ood in (False, True):
                for reduction in REDUCTIONS if _has_reduction(cls, task) else (None,):
                    yield cls, task, all_ood, reduction


def _has_reduction(cls, task) -> bool:
    criterion, _ = LOSS_INPUTS[cls](SHAPES[task])
    return hasattr(criterion, "reduction")


def _run(cls, task, all_ood, reduction, device, move_loss: bool):
    """Builds a loss and its inputs, runs ``STEPS`` steps and backpropagates the last one."""
    torch.manual_seed(0)
    criterion, args = LOSS_INPUTS[cls](SHAPES[task])
    if reduction is not None:
        criterion.reduction = reduction
    if move_loss:
        criterion.to(device)
    args = [_move(arg, device, all_ood) for arg in args]
    with warnings.catch_warnings():
        # unsupervised losses warn about the OOD samples they discard
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(STEPS):
            value = criterion(*args)
    if value.requires_grad:
        value.sum().backward()
    return criterion, args, value


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device handling tests")
class TestLossDeviceHandling(unittest.TestCase):
    def _check_on_device(self, cls, task, all_ood, reduction, move_loss: bool):
        criterion, args, value = _run(cls, task, all_ood, reduction, DEVICE, move_loss)
        self.assertEqual(value.device, DEVICE)
        self.assertTrue(torch.isfinite(value).all())
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.grad is not None:
                self.assertEqual(arg.grad.device, DEVICE)
        if move_loss:
            for name, param in criterion.named_parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.device, DEVICE, name)
        if cls not in RANDOM:
            _, _, expected = _run(cls, task, all_ood, reduction, "cpu", move_loss=False)
            torch.testing.assert_close(value.detach().cpu(), expected.detach())

    def test_to_moves_all_state(self):
        for cls in LOSSES:
            with self.subTest(cls.__name__):
                criterion, _ = LOSS_INPUTS[cls](SHAPES[next(iter(cls.info.tasks))])
                criterion.to(DEVICE)
                left = [n for n, t in _tensor_state(criterion) if t.device != DEVICE]
                self.assertEqual(left, [], "tensors that are not moved")

    def test_forward_and_backward_after_to(self):
        for cls, task, all_ood, reduction in _cases(LOSSES):
            with self.subTest(f"{cls.__name__}, {task.value}, all OOD: {all_ood}, {reduction}"):
                self._check_on_device(cls, task, all_ood, reduction, move_loss=True)

    def test_forward_and_backward_without_to(self):
        for cls, task, all_ood, reduction in _cases(c for c in LOSSES if c not in REQUIRES_TO):
            with self.subTest(f"{cls.__name__}, {task.value}, all OOD: {all_ood}, {reduction}"):
                self._check_on_device(cls, task, all_ood, reduction, move_loss=False)

    def test_vos_synthesizes_outliers(self):
        # the setup must reach the synthesis path, which holds most of the device-dependent state
        original = loss.VirtualOutlierSynthesizingRegLoss._sample_virtual_outliers
        with mock.patch.object(
            loss.VirtualOutlierSynthesizingRegLoss,
            "_sample_virtual_outliers",
            autospec=True,
            side_effect=original,
        ) as sample:
            self._check_on_device(
                loss.VirtualOutlierSynthesizingRegLoss,
                next(iter(loss.VirtualOutlierSynthesizingRegLoss.info.tasks)),
                all_ood=False,
                reduction="mean",
                move_loss=True,
            )
        self.assertGreater(sample.call_count, 0)

    def test_ii_loss_eval_mode(self):
        # in evaluation mode, the loss uses the stored running centers
        criterion, (x, y) = LOSS_INPUTS[loss.IILoss](SHAPES[next(iter(loss.IILoss.info.tasks))])
        criterion.to(DEVICE)
        criterion(x.to(DEVICE), y.to(DEVICE))
        criterion.eval()
        value = criterion(x.to(DEVICE), y.to(DEVICE))
        self.assertEqual(value.device, DEVICE)
        self.assertTrue(torch.isfinite(value))

    def test_energy_margin_update_hyperparameters(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 3).to(DEVICE)
        logistic_regression = nn.Linear(1, 1).to(DEVICE)
        # the loader yields CPU tensors, as usual
        loader = DataLoader(TensorDataset(torch.randn(5, 4), torch.zeros(5).long()), batch_size=4)
        criterion = loss.EnergyMarginLoss(full_train_loss=1.0).to(DEVICE)
        criterion.update_hyperparameters(model, loader, logistic_regression)
        self.assertEqual(criterion.lam.device, DEVICE)
        self.assertEqual(criterion.lam2.device, DEVICE)


class TestRequiresTo(unittest.TestCase):
    """``REQUIRES_TO`` lists exactly the losses that need ``.to()``, and they say so."""

    def test_losses_that_require_to_hold_tensors(self):
        # scalars mix with tensors on any device, so they alone do not require .to()
        for cls in REQUIRES_TO:
            with self.subTest(cls.__name__):
                criterion, _ = LOSS_INPUTS[cls](SHAPES[next(iter(cls.info.tasks))])
                state = [n for n, t in _tensor_state(criterion) if t.dim() > 0]
                self.assertTrue(state, "has no tensor state, so it should work without .to()")

    def test_losses_that_require_to_document_it(self):
        for cls in REQUIRES_TO:
            with self.subTest(cls.__name__):
                self.assertIn("``.to(device)``", cls.__doc__)
