"""
Multi-Layer Mahalanobis
==============================

Running :class:`MultiMahalanobis <pytorch_ood.detector.MultiMahalanobis>` on CIFAR 10.

"""

import logging

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import MultiMahalanobis
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

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)

# create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# %%
# Stage 1: Create DNN pre-trained on CIFAR 10
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()

layer1 = model.conv1
layer2 = model.block1
layer3 = model.block2
layer4 = model.block3
layer5 = nn.Sequential(model.bn1, model.relu)

# %%
# Stage 2: Create and fit model
detector = MultiMahalanobis([layer1, layer2, layer3, layer4, layer5])

print("Fitting...")
detector.fit(train_loader, device=device)

# %%
# Stage 3: Evaluate Detectors
print("Testing...")
metrics = OODMetrics()

for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())

# %%
# This produces a table with the following output:
# {'AUROC': 0.9601144790649414, 'AUPR-IN': 0.9439688324928284, 'AUPR-OUT': 0.9745389223098755, 'FPR95TPR': 0.23440000414848328}
