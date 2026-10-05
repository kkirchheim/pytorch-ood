"""
GradNorm
==========

Running :class:`GradNorm <pytorch_ood.detector.GradNorm>` on CIFAR 10.

The detector computes the :math:`\\ell_1`-norm of the gradient of the KL divergence between the
uniform distribution and the softmax output, with respect to the weights of the final classification
layer. This norm is the product of the :math:`\\ell_1`-norm of the penultimate features and the
:math:`\\ell_1`-distance of the softmax output from the uniform distribution. In-distribution inputs
tend to produce larger gradient norms, so the score is negated.

The method was proposed for large-scale models (ImageNet). On CIFAR 10, the feature norms of the
model used here are larger for the OOD data, so the detector performs poorly.
"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/detectors.png"

import logging

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import GradNorm
from pytorch_ood.metrics import OODMetrics
from pytorch_ood.model import load_model, load_transform
from pytorch_ood.utils import ToUnknown, fix_random_seed

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
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)

# %%
# Stage 2: Create the detector, which requires no fitting. Gradients are only computed for the
# weights of the final layer.
detector = GradNorm(model, param_filter=lambda name: name == "fc.weight")

# %%
# Stage 3: Evaluate
print("Testing...")

metrics = OODMetrics()
for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
