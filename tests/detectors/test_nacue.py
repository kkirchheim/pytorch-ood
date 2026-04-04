import unittest
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.detector import NACUE
from pytorch_ood.utils import OODMetrics


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
        det.to("cpu")
        det.fit(loader)

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
        det.to("cpu")
        det.fit(loader)

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
        det1.to("cpu")
        det1.fit(loader1)

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
        det2.to("cpu")
        det2.fit(loader2)
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
        det.to("cpu")
        det.fit(loader)

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
            det.to("cpu")
            det.fit(loader)

    def test_mock_performance(self):
        """
        Train TinyConvClassifier on class-discriminative images and verify that
        NACUE assigns higher outlier scores to OOD images (uniform noise) than to
        ID images (class-specific channel patterns).

        Each class has a distinct channel brightly lit; OOD images have no such
        structure. A well-trained model should be confident on ID and uncertain
        on OOD, leading to different activation coverage patterns.
        """
        torch.manual_seed(0)
        n_classes = 3
        img_size = 16

        def make_images(n_per_class, channel_value, noise_std=0.05, seed=0):
            g = torch.Generator().manual_seed(seed)
            n = n_per_class * n_classes
            x = torch.zeros(n, 3, img_size, img_size)
            y = torch.repeat_interleave(torch.arange(n_classes), n_per_class)
            for c in range(n_classes):
                sl = slice(c * n_per_class, (c + 1) * n_per_class)
                x[sl, c] = channel_value
                x[sl] += torch.randn(n_per_class, 3, img_size, img_size, generator=g) * noise_std
            return x, y

        x_train, y_train = make_images(n_per_class=120, channel_value=1.0, seed=1)
        x_id, y_id = make_images(n_per_class=30, channel_value=1.0, seed=2)

        # OOD: small-amplitude random noise — no class-specific channel pattern
        g_ood = torch.Generator().manual_seed(3)
        x_ood = torch.randn(90, 3, img_size, img_size, generator=g_ood) * 0.05
        y_ood = torch.full((90,), -1, dtype=torch.long)

        # Train the model
        model = TinyConvClassifier(num_classes=n_classes).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
        for _ in range(30):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                optimizer.step()
        model.eval()

        # Fit NACUE on training data
        detector = NACUE(
            model=model,
            layers=[model.block2, model.bn],
            m_bins=[50, 50],
            alpha=[20.0, 20.0],
            o_star=[10, 10],
            device="cpu",
        )
        fit_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)
        detector.to("cpu")
        detector.fit(fit_loader)

        # Score ID and OOD
        metrics = OODMetrics()
        with torch.enable_grad():
            metrics.update(detector(x_id), y_id)
            metrics.update(detector(x_ood), y_ood)

        results = metrics.compute()
        self.assertGreater(
            results["AUROC"], 0.70, f"Expected AUROC > 0.70, got {results['AUROC']:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
