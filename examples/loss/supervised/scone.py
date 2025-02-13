"""

Scone
-------------------------

We train a model with  :class:`Energy Margin <pytorch_ood.loss.EnergyMarginLoss>` on the CIFAR10.

We can use a model pre-trained on the :math:`32 \\times 32` resized version of the ImageNet as a foundation.
As outlier data, we use :class:`TinyImages300k <pytorch_ood.dataset.img.TinyImages300k>`, a cleaned version of the
TinyImages database, which contains random images scraped from the internet.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from typing import Callable

import numpy as np
from numpy import floating

from pytorch_ood.dataset.img import Textures, TinyImages300k
from pytorch_ood.detector import EnergyBased
from pytorch_ood.loss import EnergyMarginLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, to_np

torch.manual_seed(123)

# maximum number of epochs and training iterations
n_epochs = 100
device = "cuda:0"


def evaluate_classification_loss_training(
    model: Callable[[Tensor], Tensor], train_loader_in
) -> floating:
    """
    Evaluate classification loss on ID training dataset.

    :param model: neural network to pass inputs to
    :param train_loader_in: dataset to extract from
    :return: ndarray with average loss
    """
    model.eval()
    losses = []
    for in_set in train_loader_in:
        data = in_set[0]
        target = in_set[1]

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # forward
        y = model(data)

        # in-distribution classification accuracy
        loss_ce = F.cross_entropy(y, target, reduction="none")

        losses.extend(list(to_np(loss_ce)))

    avg_loss = np.mean(np.array(losses))
    print("average loss fr classification {}".format(avg_loss))

    return avg_loss


# %%
# Setup preprocessing and data
trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor()])

# setup ID training data
dataset_in_train = CIFAR10(root="data", train=True, download=True, transform=trans)

# setup OOD training data, use ToUnknown() to mark labels as OOD
# this way, outlier exposure can automatically decide if the training samples are ID or OOD
dataset_out_train = TinyImages300k(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# setup ID test data
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# TODO: Add Covariate-shifted Data

# setup OOD test data, use ToUnknown() to mark labels as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# create data loaders
train_loader = DataLoader(dataset_in_train + dataset_out_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

train_loader_in = DataLoader(dataset_in_train, batch_size=128)

# %%
# Create DNN, pretrained on the imagenet excluding cifar10 classes
model = WideResNet(num_classes=1000, pretrained="imagenet32-nocifar")
# we have to replace the final layer to account for the lower number of
# classes in the CIFAR10 dataset
model.fc = torch.nn.Linear(model.fc.in_features, 10)

model.to(device)

logistic_regression = nn.Linear(1, 1)

logistic_regression.to(device)

opti = SGD(
    list(model.parameters()) + list(logistic_regression.parameters()),
    lr=0.0001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
)

# Calculate ID Classification Loss Before Fine-Tuning
full_train_loss = evaluate_classification_loss_training(
    model=model, train_loader_in=train_loader_in
)

criterion = EnergyMarginLoss(full_train_loss=full_train_loss)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=opti,
    milestones=[int(n_epochs * 0.5), int(n_epochs * 0.75), int(n_epochs * 0.9)],
    gamma=0.5,
)


# %%
# Define a function to test the model
def test():
    energy = EnergyBased(model)

    metrics_energy = OODMetrics()
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            metrics_energy.update(energy(x.to(device)), y)

    print(metrics_energy.compute())
    model.train()


# %%
# Start training
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    for x, y in train_loader:
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device), logistic_regression)
        opti.zero_grad()
        loss.backward()
        opti.step()
    criterion.update_hyperparameters(
        model=model,
        train_loader_in=train_loader_in,
        logistic_regression=logistic_regression,
    )
    test()
    scheduler.step()
