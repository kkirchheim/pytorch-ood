import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.loss import EnergyMarginLoss
from src.pytorch_ood.utils import evaluate_energy_logistic_loss

torch.manual_seed(123)

logistic_regression = torch.nn.Linear(1, 1)


class TestEnergyMargin(unittest.TestCase):
    """
    Test code for margin-energy-based optimization
    """

    def test_forward(self):
        criterion = EnergyMarginLoss(full_train_loss=0)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        # At least one sample has to be OOD; otherwise, energy_loss_out returns NaN
        target[0:1] = -1

        loss = criterion(logits, target, logistic_regression)

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_backward(self):
        criterion = EnergyMarginLoss(full_train_loss=0)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        loss = criterion(logits, target, logistic_regression)
        loss.backward()

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

    def test_inccreasing_eta(self):
        criterion = EnergyMarginLoss(full_train_loss=0, eta=1.0)
        logits = torch.randn(size=(128, 10))
        target = torch.zeros(size=(128,)).long()
        target[5:] = -1

        low_eta_loss = criterion(logits, target, logistic_regression)

        self.assertIsNotNone(low_eta_loss)
        self.assertGreater(low_eta_loss, 0)

        criterion = EnergyMarginLoss(full_train_loss=0, eta=5.0)

        high_eta_loss = criterion(logits, target, logistic_regression)

        self.assertIsNotNone(high_eta_loss)
        self.assertGreater(high_eta_loss, 0)

        # introducing wider margin should minimize the loss further
        self.assertGreater(low_eta_loss, high_eta_loss)

    def test_batch_without_id_or_ood_raises(self):
        # otherwise the loss is NaN, or the constraints silently have no effect
        criterion = EnergyMarginLoss(full_train_loss=1.0)
        for target in (torch.zeros(8).long(), -torch.ones(8).long()):
            with self.subTest(target=target[0].item()):
                with self.assertRaises(ValueError):
                    criterion(torch.randn(8, 10), target, logistic_regression)


def _old_evaluate_energy_logistic_loss(model, train_loader_in, logistic_regression):
    # the implementation before the device fix, which only worked with the model on cuda:0
    import numpy as np
    import torch.nn.functional as F

    model.eval()
    sigmoid, logistic, ce = [], [], []
    for data, target in train_loader_in:
        data, target = data.cuda(), target.cuda()
        y = model(data)
        Ec_in = torch.logsumexp(y, dim=1)
        labels = torch.ones(len(data)).cuda()
        logistic.extend(
            F.binary_cross_entropy_with_logits(
                logistic_regression(Ec_in.unsqueeze(1)).squeeze(), labels, reduction="none"
            )
            .data.cpu()
            .numpy()
        )
        sigmoid.extend(
            torch.sigmoid(logistic_regression(Ec_in.unsqueeze(1)).squeeze()).data.cpu().numpy()
        )
        ce.extend(F.cross_entropy(y, target, reduction="none").data.cpu().numpy())
    return np.mean(sigmoid), np.mean(logistic), np.mean(ce)


def _loader(n: int, batch_size: int = 4) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, 4), torch.zeros(n).long()), batch_size=batch_size
    )


class TestEvaluateEnergyLogisticLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = torch.nn.Linear(4, 3)
        self.phi = torch.nn.Linear(1, 1)

    def test_cpu(self):
        # the model and the data stay on the CPU, also when CUDA is available
        results = evaluate_energy_logistic_loss(self.model, _loader(8), self.phi, device="cpu")
        self.assertTrue(all(np.isfinite(r) for r in results))

    def test_last_batch_of_size_one(self):
        results = evaluate_energy_logistic_loss(self.model, _loader(5), self.phi)
        self.assertTrue(all(np.isfinite(r) for r in results))

    def test_matches_batchwise_computation(self):
        loader = _loader(9)
        x, y = loader.dataset.tensors
        logits = self.model(x).detach()
        energy = torch.logsumexp(logits, dim=1, keepdim=True)
        expected_sigmoid = torch.sigmoid(self.phi(energy)).mean().item()
        expected_ce = torch.nn.functional.cross_entropy(logits, y).item()
        sigmoid, _, ce = evaluate_energy_logistic_loss(self.model, loader, self.phi)
        self.assertAlmostEqual(sigmoid, expected_sigmoid, places=6)
        self.assertAlmostEqual(ce, expected_ce, places=6)

    @unittest.skipUnless(torch.cuda.is_available(), "the old implementation required CUDA")
    def test_identical_to_old_implementation(self):
        # batch size divides the data, the only case the old implementation handled
        model, phi = self.model.cuda(), self.phi.cuda()
        loader = _loader(8)
        old = _old_evaluate_energy_logistic_loss(model, loader, phi)
        new = evaluate_energy_logistic_loss(model, loader, phi, device="cuda")
        self.assertEqual(tuple(old), tuple(new))

    def test_update_hyperparameters_on_cpu(self):
        criterion = EnergyMarginLoss(full_train_loss=1.0)
        criterion.update_hyperparameters(self.model, _loader(5), self.phi)
        self.assertTrue(torch.isfinite(criterion.lam))

    def test_without_gradients_in_eval_mode(self):
        grad_enabled = []
        self.model.register_forward_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))
        self.model.train()
        evaluate_energy_logistic_loss(self.model, _loader(8), self.phi)
        self.assertEqual(grad_enabled, [False, False])
        self.assertFalse(self.model.training)

    def test_state_in_state_dict(self):
        criterion = EnergyMarginLoss(full_train_loss=1.0)
        criterion.lam.fill_(0.5)
        criterion.lam2.fill_(0.25)
        criterion.in_constraint_weight.fill_(3.0)
        restored = EnergyMarginLoss(full_train_loss=1.0)
        restored.load_state_dict(criterion.state_dict())
        self.assertEqual(restored.lam.item(), 0.5)
        self.assertEqual(restored.lam2.item(), 0.25)
        self.assertEqual(restored.in_constraint_weight.item(), 3.0)
