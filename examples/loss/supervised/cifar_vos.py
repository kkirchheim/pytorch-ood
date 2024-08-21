"""
Virtual  Outlier Synthesizer Loss
-------------------------------------

We train a model with :class:`Virtual Outlier Synthesizer Loss<pytorch_ood.loss.vos.VirtualOutlierSynthesizingRegLoss>` on the CIFAR10.

We then use the :class:`WeightedEBO<pytorch_ood.detector.WeightedEBO>` OOD detector.

We can use a model pre-trained on the :math:`32 \\times 32` resized version of the ImageNet as a foundation.
As outlier data, we use :class:`TinyImages300k <pytorch_ood.dataset.img.TinyImages300k>`, a cleaned version of the
TinyImages database, which contains random images scraped from the internet.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import EnergyBased, WeightedEBO
from pytorch_ood.loss import VirtualOutlierSynthesizingRegLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

device = "cuda:5"
batch_size = 128
num_epochs = 10
lr = 0.1
num_classes = 10

fix_random_seed(12345)
g = torch.Generator()
g.manual_seed(0)


# %%
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# %%
# Setup datasets, train on cifar.
trans = WideResNet.transform_for("cifar10-pt")

dataset = CIFAR10(root="data", train=True, transform=trans, download=True)
# dataset_test = CIFAR100(root="data", transform=trans, target_transform=ToUnknown(), download=True)
# setup IN test data
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
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        num_epochs * len(loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / lr,
    ),
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
detector = WeightedEBO(model, weights_energy)
detector1 = EnergyBased(model)
metrics = OODMetrics()
metrics1 = OODMetrics()

with torch.no_grad():
    for n, (x, y) in enumerate(test_loader):
        y, x = y.to(device), x.to(device)
        y_hat = model(x)
        o = detector.predict_features(y_hat)
        o1 = detector1.predict_features(y_hat)

        metrics.update(o, y)
        metrics1.update(o1, y)
        if n % 10 == 0:
            print(f"Epoch {epoch:03d} [{n:05d}/{len(test_loader):05d}] ")

print(f"WeightedEBO: {metrics.compute()}")
print(f"EnergyBased: {metrics1.compute()}")
# %%
# Output:
# {'AUROC': 0.8593780398368835, 'AUPR-IN': 0.744148850440979, 'AUPR-OUT': 0.9147135615348816, 'FPR95TPR': 0.4684000015258789}
