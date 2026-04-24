"""
PNML
==============================

Comparing :class:`PNML <pytorch_ood.detector.PNML>`,
:class:`MaxSoftmax <pytorch_ood.detector.MaxSoftmax>`, and
:class:`EnergyBased <pytorch_ood.detector.EnergyBased>` on a small CIFAR-10 benchmark.

This example mirrors the other detector demos, but keeps fitting and evaluation on
small subsets so it stays reasonably fast.
"""

import logging

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import EnergyBased, MaxSoftmax, PNML
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)

fix_random_seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128

# Use subsets to keep this example lightweight while still running a real benchmark.
n_train = 1_000
n_in_test = 1_000
n_out_test = 1_000

# %%
# Setup preprocessing and data
trans = WideResNet.transform_for("cifar10-pt")

dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

train_loader = DataLoader(
    Subset(dataset_train, range(n_train)),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    Subset(dataset_in_test, range(n_in_test)) + Subset(dataset_out_test, range(n_out_test)),
    batch_size=batch_size,
)


def evaluate(name, detector, loader):
    metrics = OODMetrics()
    for x, y in loader:
        metrics.update(detector(x.to(device)), y)

    print(f"{name}:")
    print(metrics.compute())


# %%
# Stage 1: Create DNN pre-trained on CIFAR-10
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()

# %%
# Stage 2: Create and fit detectors
pnml = PNML(model.features, model.fc).to(device)
msp = MaxSoftmax(model).to(device)
energy = EnergyBased(model).to(device)

print("Fitting...")
pnml.fit(train_loader)

# %%
# Stage 3: Evaluate detectors
print("Testing...")

evaluate("PNML", pnml, test_loader)
evaluate("MSP", msp, test_loader)
evaluate("Energy", energy, test_loader)
