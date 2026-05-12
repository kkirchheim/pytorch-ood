"""
Example: CUB_SSB Benchmark with OOD Detection

This demonstrates the Semantic Split Benchmark (SSB) on CUB-200-2011,
evaluating OOD detectors on fine-grained open-set recognition.

:see Paper: Dissecting Out-of-Distribution Detection and Open-Set Recognition
    https://arxiv.org/abs/2408.16757
"""

import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from pytorch_ood.benchmark import CUB_SSB
from pytorch_ood.utils import oscr_score

# ─── Setup ────────────────────────────────────────────────────────────────────

# Path where CUB-200-2011 data will be stored
DATA_ROOT = "./data"

# Simple transform: resize to 224x224 and normalize with ImageNet stats
transform = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ─── Create Benchmark ────────────────────────────────────────────────────────

# Initialize the benchmark. Use download=True on first run.
# On first call, this downloads and caches the SSB class splits.
benchmark = CUB_SSB(root=DATA_ROOT, transform=transform, download=False)

print(f"✓ Benchmark created: {benchmark.__class__.__name__}")
print(f"  - Train set size: {len(benchmark.train_set())}")
print(f"  - OOD split names: {benchmark.ood_names}")

# ─── Create Dummy Detector for Demo ────────────────────────────────────────

# In practice, you'd train a classifier and wrap its logits/features
# For this example, we create a dummy detector that outputs random scores


class DummyDetector(torch.nn.Module):
    """Dummy detector that outputs random outlier scores."""

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.randn(batch_size)


detector = DummyDetector()

# ─── Evaluate ──────────────────────────────────────────────────────────────

# Evaluate on the benchmark (small batch for demo)
# Each test_set combines ID and OOD samples: [Easy OOD set], [Hard OOD set]
loader_kwargs = {"batch_size": 32, "num_workers": 0}

print("\n✓ Evaluating detector...")
results = benchmark.evaluate(detector, loader_kwargs=loader_kwargs, device="cpu")

for i, result in enumerate(results):
    ood_name = benchmark.ood_names[i]
    print(f"  OOD Condition: {ood_name}")
    print(f"    AUROC:     {result['AUROC']:.4f}")
    print(f"    FPR@95TPR: {result['FPR95TPR']:.4f}")

# ─── OSCR Metric (Open-Set Classification Rate) ────────────────────────────

# OSCR requires model predictions (not just OOD scores).
# It measures joint: (1) correct classification of known samples, (2) detection of unknown.
#
# In a real scenario, you'd have:
#   - A classifier that outputs logits/predictions
#   - An OOD detector that outputs outlier scores
#   - Evaluate OSCR using both

print("\n✓ OSCR Metric Example (with synthetic data)")

# Synthetic example: 100 known samples (all correctly classified),
# 50 unknown samples, perfect OOD detection.
scores_known = torch.zeros(100)  # ID samples have low scores
scores_ood = torch.ones(50)  # OOD samples have high scores
scores = torch.cat([scores_known, scores_ood])

predictions_known = torch.arange(100)  # Predicted classes match true classes
predictions_ood = torch.zeros(50, dtype=torch.long)  # Doesn't matter for OOD
predictions = torch.cat([predictions_known, predictions_ood])

labels_known = torch.arange(100)  # True classes
labels_ood = torch.full((50,), -1)  # -1 marks unknown/OOD
labels = torch.cat([labels_known, labels_ood])

oscr = oscr_score(scores, predictions, labels)
print(f"  Perfect detector: OSCR = {oscr:.4f} (expected ≈ 1.0)")

print("\n✓ Done! For real evaluation, provide a trained model to the detector.")
