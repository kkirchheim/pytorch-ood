"""
Monte Carlo Dropout
==============================

Uses MCD based on a pre-trained model from the Hendrycks baseline paper.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import MCD
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

fix_random_seed(123)
device = "cuda:0"

trans = WideResNet.transform_for("cifar10-pt")


# setup ID test data
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
# setup OOD test data
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# merge dataset and create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# Stage 1: Create DNN
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device)

# Stage 2: Create Detector
detector = MCD(model, samples=30)

# Stage 3: Evaluate Detectors
metrics = OODMetrics()

for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
