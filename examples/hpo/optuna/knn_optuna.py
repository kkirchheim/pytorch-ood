"""
Tuning KNN with Optuna (TPE)
============================

Drives :class:`KNN <pytorch_ood.detector.KNN>` from Optuna's TPE sampler instead of the
built-in grid search. The detector is fitted once on the CIFAR-10 training set; each
trial only changes ``k`` (which does not affect the fitted index) and re-scores the
validation set. Ranges are defined by hand in the objective with ``trial.suggest_*``.
"""

import logging

import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import KNN
from pytorch_ood.model import load_model, load_transform
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)
fix_random_seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
n_fit = 2_000
n_val = 1_000

# %%
# Model and data: fit on CIFAR-10 train (ID), validate on CIFAR-10 test (ID) + Textures (OOD).
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)
trans = load_transform("wrn-40-2/cifar10/crossentropy")

cifar_train = CIFAR10(root="data", train=True, download=True, transform=trans)
cifar_test = CIFAR10(root="data", train=False, download=True, transform=trans)
textures = Textures(root="data", download=True, transform=trans, target_transform=ToUnknown())

# %%
# Fit once on training data (``KNN.fit`` extracts features internally). Since ``k`` is a
# query-time parameter, the nearest-neighbor index is reused across all trials.
detector = KNN(model.features)
detector.fit(DataLoader(Subset(cifar_train, range(n_fit)), batch_size=batch_size))

val_loader = DataLoader(
    Subset(cifar_test, range(n_val)) + Subset(textures, range(n_val)), batch_size=batch_size
)


# %%
# Objective: TPE over ``k``, with the range defined by hand.
def objective(trial):
    detector.set_hyperparameters(k=trial.suggest_int("k", 1, 500, log=True))
    metric = OODMetrics()
    with torch.no_grad():
        for x, y in val_loader:
            metric.update(detector(x.to(device)), y)
    return metric.compute()["AUROC"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=False)

# %%
# Results: the best validation AUROC found for each value of ``k`` that TPE tried.
results = study.trials_dataframe(attrs=("params", "value")).rename(
    columns={"params_k": "k", "value": "AUROC"}
)
table = results.groupby("k")["AUROC"].max().reset_index().sort_values("AUROC", ascending=False)
print(table.to_string(index=False))
print(f"\nBest: {study.best_params}  AUROC = {study.best_value:.4f}")

# %%
# This prints the best validation AUROC per ``k``. TPE concentrates its trials on small
# ``k`` (where KNN performs best here) and quickly abandons large ``k`` (selected rows):
#
# +-----+--------+
# | k   | AUROC  |
# +=====+========+
# | 3   | 0.9164 |
# +-----+--------+
# | 5   | 0.9159 |
# +-----+--------+
# | 7   | 0.9142 |
# +-----+--------+
# | 14  | 0.9106 |
# +-----+--------+
# | 27  | 0.9057 |
# +-----+--------+
# | 43  | 0.9015 |
# +-----+--------+
# | 125 | 0.8875 |
# +-----+--------+
# | 236 | 0.2343 |
# +-----+--------+
