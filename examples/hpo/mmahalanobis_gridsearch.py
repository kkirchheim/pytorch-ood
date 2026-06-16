"""
Tuning Multi-Layer Mahalanobis with GridSearch
===============================================

:class:`MultiMahalanobis <pytorch_ood.detector.MultiMahalanobis>` combines per-layer
Mahalanobis distances using one weight ``alpha`` **per layer**. Because the number of
weights equals the number of layers, the search space cannot be a fixed class attribute
like for other detectors — it has to be **constructed at runtime** from the chosen layers.

This example builds the search space manually (comparing uniform weighting against using
each layer on its own), assigns it to the detector, and selects the best weighting on a
held-out validation set with :class:`GridSearch <pytorch_ood.utils.GridSearch>`.
"""

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import MultiMahalanobis
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import GridSearch, OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)
fix_random_seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128

# Use subsets to keep this example lightweight while still running a real benchmark.
n_fit = 1_000
n_val = 1_000
n_test = 1_000

# %%
# Set up data: an ID **fit** set, and ID + OOD **validation** and **test** sets.
trans = WideResNet.transform_for("cifar10-pt")

cifar_train = CIFAR10(root="data", train=True, download=True, transform=trans)
cifar_test = CIFAR10(root="data", train=False, download=True, transform=trans)
textures = Textures(root="data", download=True, transform=trans, target_transform=ToUnknown())

fit_loader = DataLoader(Subset(cifar_train, range(n_fit)), batch_size=batch_size)

val_dataset = Subset(cifar_test, range(n_val)) + Subset(textures, range(n_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Subset(cifar_test, range(n_val, n_val + n_test)) + Subset(
    textures, range(n_val, n_val + n_test)
)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
# Decompose the model into a list of layers; each contributes one Mahalanobis score.
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)
layers = [
    model.conv1,
    model.block1,
    model.block2,
    model.block3,
    nn.Sequential(model.bn1, model.relu),
]
detector = MultiMahalanobis(layers)

# %%
# Build the search space manually. ``alpha`` is a vector with one weight per layer, so its
# shape depends on ``len(layers)`` — we cannot hard-code it on the class. Here we compare
# uniform weighting (all layers) against using each individual layer on its own.
n_layers = len(layers)
alpha_candidates = [[1.0] * n_layers]  # uniform: use all layers
for i in range(n_layers):
    one_hot = [0.0] * n_layers
    one_hot[i] = 1.0
    alpha_candidates.append(one_hot)  # use only layer i

detector.hyperparameter_space = {"alpha": alpha_candidates}

# %%
# Select the best weighting on the validation set.
search = GridSearch(detector, fit_loader=fit_loader, val_loader=val_loader, device=device)
best = search.run()

print(f"\nBest alpha: {best['alpha']} (validation AUROC = {search.best_score_:.4f})")
for record in search.results_:
    print(f"  alpha={record['params']['alpha']}   validation AUROC = {record['score']:.4f}")

# %%
# ``search.run()`` leaves the detector configured with the best weighting, so we can
# evaluate it directly on the held-out test set.
metrics = OODMetrics()
with torch.no_grad():
    for x, y in test_loader:
        metrics.update(detector(x.to(device)), y)

print(f"\nTest metrics with tuned alpha={detector.alpha}:")
print({k: round(v, 4) for k, v in metrics.compute().items()})
