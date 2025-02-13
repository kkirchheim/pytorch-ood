"""
Gram
==============================

Running :class:`Gram <pytorch_ood.detector.Gram>` on CIFAR 10.

"""

import logging

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import torch.nn.functional as F
from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import Gram
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)

fix_random_seed(123)

device = "cuda"

# %%
# Setup preprocessing and data
trans = WideResNet.transform_for("cifar10-pt")

dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=10)

# create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128, num_workers=10)

# %%
# Stage 1: Create DNN pre-trained on CIFAR 10
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()

layer1 = model.conv1
layer2 = model.block1
layer3 = model.block2
layer4 = model.block3
layer5 = nn.Sequential(model.bn1, model.relu)

head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc)

# %%
# Stage 2: Create and fit model
detector = Gram(
    head,
    [layer1, layer2, layer3, layer4, layer5],
    num_classes=10,
    num_poles_list=[1, 2, 3, 4, 5],
)

print("Fitting...")
detector.fit(train_loader, device=device)


# %%
# Stage 3: Evaluate Detectors
print("Testing...")

metrics = OODMetrics()
for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)


print(metrics)

# %%
# This produces a table with the following output:

# {'AUROC': 0.8175439834594727, 'AUTC': 0.4554872214794159, 'AUPR-IN': 0.8401336073875427, 'AUPR-OUT': 0.7695250511169434, 'FPR95TPR': 0.8087999820709229}
