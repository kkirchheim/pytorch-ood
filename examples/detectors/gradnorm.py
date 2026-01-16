"""
GradNorm
==============================

Running :class:`GradNorm <pytorch_ood.detector.GradNorm>` on CIFAR 10.

"""

import logging

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import GradNorm
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

model.requires_grad_(False)
model.fc.requires_grad_(True)


# %%

# Stage 2: Create detector, fitting is not required
detector = GradNorm(model, param_filter=lambda name: name.startswith("fc"))


# %%
# Stage 3: Evaluate Detectors
print("Testing...")

metrics = OODMetrics()
for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)


print(metrics.compute())

# %%
# This produces the following output:
# {'AUROC': 0.4999113380908966, 'AUTC': 0.5440057516098022, 'AUPR-IN': 0.31969308853149414, 'AUPR-OUT': 0.6802297830581665, 'FPR95TPR': 1.0}
