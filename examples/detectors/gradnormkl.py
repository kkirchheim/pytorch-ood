"""
GradNormKL
==============================

Running :class:`GradNormKL <pytorch_ood.detector.GradNormKL>` on CIFAR 10.

This method computes the :math:`\\ell_1`-norm of the gradient of the KL divergence between the
softmax output and a uniform distribution, with respect to the weights of the final classification
head. In-distribution inputs tend to produce more peaked predictions (larger divergence from
uniform), yielding higher gradient norms. The score is negated so that higher values indicate OOD.

Unlike :class:`GradNorm <pytorch_ood.detector.GradNorm>`, no labeled OOD data or additional
classifier is required.

"""

import logging

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import GradNormKL
from pytorch_ood.model import WideResNet, load_model, load_transform
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)

fix_random_seed(123)

device = "cuda"

# %%
# Setup preprocessing and data
trans = load_transform("wrn-40-2/cifar10/crossentropy")

dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=64, num_workers=4)

# %%
# Stage 1: Load pre-trained WideResNet for CIFAR-10.
# Disable gradients for the backbone; only the final FC layer needs them.
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)

model.requires_grad_(False)
model.fc.requires_grad_(True)

# %%
# Stage 2: Create detector — no fitting required.
detector = GradNormKL(model, param_filter=lambda name: name.startswith("fc"))

# %%
# Stage 3: Evaluate
print("Testing...")

metrics = OODMetrics()
for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
