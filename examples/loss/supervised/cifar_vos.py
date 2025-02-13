"""
Virtual  Outlier Synthesizer Loss
-------------------------------------

We train a model with :class:`Virtual Outlier Synthesizer Loss<pytorch_ood.loss.vos.VirtualOutlierSynthesizingRegLoss>` on the CIFAR10.

We then use the :class:`WeightedEBO<pytorch_ood.detector.WeightedEBO>` OOD detector.

We can use a model pre-trained on the :math:`32 \\times 32` resized version of the ImageNet as a foundation.
"""

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import EnergyBased, WeightedEBO
from pytorch_ood.loss import VirtualOutlierSynthesizingRegLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

device = "cuda:0"
batch_size = 128
num_epochs = 10
lr = 0.1
num_classes = 10

fix_random_seed(12345)
g = torch.Generator()
g.manual_seed(0)


# %%
# Setup datasets, train on cifar.
trans = WideResNet.transform_for("cifar10-pt")

dataset = CIFAR10(root="data", train=True, transform=trans, download=True)

# setup ID test data
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# setup OOD test data, use ToUnknown() to mark labels as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)


loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    worker_init_fn=fix_random_seed,
    generator=g,
)

# %%
# Setup model
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device)

# %%
# Create neural network functions (layers)
phi = torch.nn.Linear(1, 2).to(device)
weights_energy = torch.nn.Linear(num_classes, 1).to(device)
torch.nn.init.uniform_(weights_energy.weight)

criterion = VirtualOutlierSynthesizingRegLoss(
    phi,
    weights_energy,
    device=device,
    num_classes=num_classes,
    num_input_last_layer=128,
    fc=model.fc,
    sample_number=10000,
    select=1,
    sample_from=1000,
    alpha=0.1,
)

# %%
# Train model for some epochs
optimizer = torch.optim.SGD(
    list(model.parameters()) + list(phi.parameters()) + list(weights_energy.parameters()),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4,
)


# setup scheduler for optimizer (recommended)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * len(loader),
)
loss_ema = 0

for epoch in range(num_epochs):
    for n, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        y, x = y.to(device), x.to(device)

        features = model.features(x)
        y_hat = model.fc(features)
        loss = criterion(y_hat, features, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_ema = 0.8 * loss_ema + 0.2 * loss.item()

        if n % 10 == 0:
            print(f"Epoch {epoch:03d} [{n:05d}/{len(loader):05d}] \t Loss: {loss_ema:02.2f}")

# %%
# Evaluate
print("Evaluating")
model.eval()
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=64)
detector_weightedEBO = WeightedEBO(model, weights_energy)
detector_energyBased = EnergyBased(model)
metrics_weightedEBO = OODMetrics()
metrics_energyBased = OODMetrics()

with torch.no_grad():
    for n, (x, y) in enumerate(test_loader):
        y, x = y.to(device), x.to(device)
        y_hat = model(x)
        o = detector_weightedEBO.predict_features(y_hat)
        o1 = detector_energyBased.predict_features(y_hat)

        metrics_weightedEBO.update(o, y)
        metrics_energyBased.update(o1, y)
        if n % 10 == 0:
            print(f"Epoch {epoch:03d} [{n:05d}/{len(test_loader):05d}] ")

print(f"WeightedEBO: {metrics_weightedEBO.compute()}")
print(f"EnergyBased: {metrics_energyBased.compute()}")
# %%
# Output:
# WeightedEBO: {'AUROC': 0.9192541837692261, 'AUPR-IN': 0.8389347195625305, 'AUPR-OUT': 0.954131007194519, 'FPR95TPR': 0.2897000014781952}
# EnergyBased: {'AUROC': 0.9227883815765381, 'AUPR-IN': 0.8493221998214722, 'AUPR-OUT': 0.956129789352417, 'FPR95TPR': 0.2799000144004822}
