"""
Tuning KNN with GridSearch
==========================

Demonstrates OpenOOD-style hyperparameter optimization with
:class:`GridSearch <pytorch_ood.utils.GridSearch>`.

The number of neighbors :math:`k` of the
:class:`KNN <pytorch_ood.detector.KNN>` detector is selected by grid search on a
held-out **validation** set that contains both in-distribution (CIFAR-10) and
out-of-distribution (Textures) data. The value that maximizes the validation
AUROC is then evaluated on a disjoint **test** set.

Since ``KNN`` exposes a feature encoder, ``GridSearch`` extracts the features
once and reuses them across all candidate values of :math:`k`.
"""

import logging

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import KNN
from pytorch_ood.model import load_model, load_transform
from pytorch_ood.utils import GridSearch, OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)
fix_random_seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128

# Use subsets to keep this example lightweight while still running a real benchmark.
n_fit = 2_000
n_val = 1_000
n_test = 1_000

# %%
# Set up data. We build three disjoint splits:
#
# * a **fit** set of in-distribution features for the nearest-neighbor index,
# * a **validation** set (ID + OOD) used to select ``k``,
# * a **test** set (ID + OOD) used for the final, unbiased evaluation.
trans = load_transform("wrn-40-2/cifar10/crossentropy")

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
# Create the model and detector.
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)
detector = KNN(model.features)

# %%
# Run the grid search over ``k`` on the validation set.
search = GridSearch(
    detector,
    fit_loader=fit_loader,
    val_loader=val_loader,
    hyperparameter_space={"k": [1, 5, 10, 50, 100]},
    device=device,
)
best = search.run()

print(f"\nBest hyperparameters: {best} (validation AUROC = {search.best_score_:.4f})")
for record in search.results_:
    print(f"  k={record['params']['k']:>4}   validation AUROC = {record['score']:.4f}")

# %%
# ``search.run()`` leaves the detector configured with the best ``k``, so we can
# evaluate it directly on the held-out test set.
metrics = OODMetrics()
with torch.no_grad():
    for x, y in test_loader:
        metrics.update(detector(x.to(device)), y)

print(f"\nTest metrics with tuned k={detector.k}:")
print({k: round(v, 4) for k, v in metrics.compute().items()})
