"""
NNGuide
=======

Running :class:`NNGuide <pytorch_ood.detector.NNGuide>` on CIFAR 10.

The detector builds a bank of normalized training features, each scaled by the energy of its
sample. For a test input, the guidance is the mean of the :math:`k` largest inner products between
its normalized features and the bank. The score is the product of the guidance and the energy of
the input, negated so that higher values indicate OOD.
"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/detectors.png"

import logging

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import NNGuide
from pytorch_ood.metrics import OODMetrics
from pytorch_ood.model import load_model, load_transform
from pytorch_ood.utils import ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)

fix_random_seed(123)

device = "cuda"

# %%
# Setup preprocessing and data
trans = load_transform("wrn-40-2/cifar10/crossentropy")

dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

train_loader = DataLoader(dataset_train, batch_size=256, num_workers=4)
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=256, num_workers=4)

# %%
# Stage 1: Load pre-trained WideResNet for CIFAR-10.
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)

# %%
# Stage 2: Create the detector and build the feature bank from the training data.
detector = NNGuide(encoder=model.features, head=model.fc, k=10)
detector.fit(train_loader, device=device)

# %%
# Stage 3: Evaluate
print("Testing...")

metrics = OODMetrics()
for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
