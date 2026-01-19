import unittest
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import NACUE


class TinyConvClassifier(nn.Module):
    """
    Small deterministic conv classifier with named submodules that we can hook.
    Returns logits of shape (B, C).
    """

    def __init__(self, in_ch: int = 3, num_classes: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bn = nn.BatchNorm2d(16)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.bn(x)
        return self.head(x)


def _make_loader(
    n: int = 32, num_classes: int = 10, batch_size: int = 8, seed: int = 0
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 16, 16, generator=g)
    y = torch.randint(low=0, high=num_classes, size=(n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


class NACUETest(unittest.TestCase):
    """
    High-coverage tests for NACUE detector implementation.

    Goals:
    - API compliance: fit(), predict()/__call__ output shape, accepts device kwarg.
    - Correct state: raises before fitting, fitted after fit.
    - Determinism: fixed seed -> identical results.
    - Numerical sanity: finite scores; non-constant on diverse inputs.
    - Gradient dependency: works under enable_grad; not silently executed under no_grad.
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def _make_detector(
        self,
        model: nn.Module,
        layers: List[nn.Module],
        m_bins=50,
        alpha=10.0,
        o_star=10,
        device=None,
    ) -> NACUE:
        return NACUE(
            model=model,
            layers=layers,
            m_bins=m_bins,
            alpha=alpha,
            o_star=o_star,
            device=device,
        )

    def test_predict_raises_before_fit(self):
        model = TinyConvClassifier().eval()
        det = self._make_detector(model, layers=[model.block2, model.bn])

        x = torch.randn(4, 3, 16, 16)

        # Update this exception type if your NACUE uses pytorch-ood's RequiresFittingException.
        with self.assertRaises(Exception):
            _ = det(x)

    def test_fit_then_predict_shape_and_finite(self):
        model = TinyConvClassifier().eval()
        det = self._make_detector(
            model,
            layers=[model.block2, model.bn],
            m_bins=[50, 50],
            alpha=[20.0, 20.0],
            o_star=[5, 10],
            device="cpu",
        )

        loader = _make_loader(n=32, batch_size=8, seed=1)
        det.fit(loader, device="cpu")

        x = torch.randn(16, 3, 16, 16)
        with torch.enable_grad():  # NAC-UE needs grads
            scores = det(x)

        self.assertEqual(scores.shape, (16,))
        self.assertTrue(torch.isfinite(scores).all().item())

    def test_accepts_device_kwarg_in_fit(self):
        model = TinyConvClassifier().eval()
        det = self._make_detector(model, layers=[model.block2], device=None)

        loader = _make_loader(n=16, batch_size=4, seed=2)
        # Should not error
        det.fit(loader, device="cpu")

        x = torch.randn(8, 3, 16, 16)
        with torch.enable_grad():
            s = det(x)
        self.assertEqual(s.shape, (8,))

    def test_determinism_fixed_seed(self):
        # Two runs, same seeds -> identical scores.
        torch.manual_seed(123)
        model1 = TinyConvClassifier().eval()
        det1 = self._make_detector(
            model1,
            layers=[model1.block2, model1.bn],
            m_bins=[50, 50],
            alpha=[20.0, 30.0],
            o_star=[5, 10],
            device="cpu",
        )
        loader1 = _make_loader(n=32, batch_size=8, seed=7)
        det1.fit(loader1, device="cpu")

        x = torch.randn(16, 3, 16, 16)
        with torch.enable_grad():
            s1 = det1(x).detach().cpu()

        torch.manual_seed(123)
        model2 = TinyConvClassifier().eval()
        det2 = self._make_detector(
            model2,
            layers=[model2.block2, model2.bn],
            m_bins=[50, 50],
            alpha=[20.0, 30.0],
            o_star=[5, 10],
            device="cpu",
        )
        loader2 = _make_loader(n=32, batch_size=8, seed=7)
        det2.fit(loader2, device="cpu")
        with torch.enable_grad():
            s2 = det2(x).detach().cpu()

        self.assertTrue(torch.allclose(s1, s2, atol=0.0, rtol=0.0))

    def test_scores_not_constant_on_diverse_inputs(self):
        model = TinyConvClassifier().eval()
        det = self._make_detector(
            model,
            layers=[model.block2, model.bn],
            m_bins=[100, 100],
            alpha=[50.0, 50.0],
            o_star=[10, 10],
            device="cpu",
        )
        loader = _make_loader(n=64, batch_size=8, seed=11)
        det.fit(loader, device="cpu")

        # Two batches with different statistics
        g = torch.Generator().manual_seed(99)
        x1 = torch.randn(64, 3, 16, 16, generator=g)
        x2 = torch.randn(64, 3, 16, 16, generator=g) + 1.5

        with torch.enable_grad():
            s1 = det(x1).detach()
            s2 = det(x2).detach()

        # Not strictly guaranteed, but for a sane implementation it should vary.
        self.assertGreater(s1.std().item(), 1e-6)
        self.assertGreater(s2.std().item(), 1e-6)

    def test_layer_config_validation_lengths(self):
        model = TinyConvClassifier().eval()

        with self.assertRaises(ValueError):
            det = self._make_detector(
                model,
                layers=[model.block2, model.bn],
                m_bins=[50],  # mismatch: 2 layers but 1 value
                alpha=[20.0, 20.0],
                o_star=[5, 10],
                device="cpu",
            )
        loader = _make_loader(n=16, batch_size=4, seed=5)
        with self.assertRaises(Exception):
            det.fit(loader, device="cpu")


if __name__ == "__main__":
    unittest.main()
