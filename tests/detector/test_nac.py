from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.detector import NACUE
from pytorch_ood.metrics import OODMetrics


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


class TestNACUE:
    def setup_method(self) -> None:
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

    def test_layer_config_validation_lengths(self):
        model = TinyConvClassifier().eval()

        with pytest.raises(ValueError):
            self._make_detector(
                model,
                layers=[model.block2, model.bn],
                m_bins=[50],  # mismatch: 2 layers but 1 value
                alpha=[20.0, 20.0],
                o_star=[5, 10],
                device="cpu",
            )

    def test_failed_refit_leaves_the_detector_unfitted(self):
        model = TinyConvClassifier().eval()
        detector = self._make_detector(model, [model.block2, model.bn])
        x = torch.randn(4, 3, 16, 16)
        detector.fit(DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=2))

        with pytest.raises(ValueError):
            detector.fit(DataLoader(TensorDataset(x, -torch.ones(4, dtype=torch.long))))

        with pytest.raises(RequiresFittingException):
            detector(x)

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
        assert results["AUROC"] > 0.7, f"Expected AUROC > 0.70, got {results['AUROC']:.4f}"
