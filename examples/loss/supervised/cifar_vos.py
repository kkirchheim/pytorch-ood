"""
Virtual  Outlier Synthesizer Loss
-------------------------------------

We train a model with class:´VIRTUALOUTLIERSYNTHESIZER´ on the CIFAR10.

We then use the :class:`WeightedEBO<pytorch_ood.detector.WeightedEBO>` OOD detector.

We can use a model pre-trained on the :math:`32 \\times 32` resized version of the ImageNet as a foundation.
As outlier data, we use :class:`TinyImages300k <pytorch_ood.dataset.img.TinyImages300k>`, a cleaned version of the
TinyImages database, which contains random images scraped from the internet.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad, to_tensor
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from pytorch_ood.detector import WeightedEBO
from pytorch_ood.loss import VIRTUALOUTLIERSYNTHESIZER, CrossEntropyLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, fix_random_seed, ToUnknown
from torchvision.datasets import CIFAR10,CIFAR100



device = "cuda:0"
batch_size = 256*4
num_epochs = 20
lr = 0.0001
num_classes = 10 

fix_random_seed(12345)
g = torch.Generator()
g.manual_seed(0)


# %%
# Setup preprocessing
preprocess_input = get_preprocessing_fn("resnet50", pretrained="imagenet")


def my_transform(img, target):
    img = to_tensor(img)[:3, :, :]  # drop 4th channel
    img = torch.moveaxis(img, 0, -1)
    img = preprocess_input(img)
    img = torch.moveaxis(img, -1, 0)

    return img.float(), target


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# %%
# Setup datasets, train on cifar.
trans = WideResNet.transform_for("cifar10-pt")

dataset = CIFAR10(root="data", train=False, transform=trans, download=True)
dataset_test = CIFAR100(root="data", transform=trans, target_transform=ToUnknown(), download=True)



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
model = WideResNet(
    num_classes=num_classes,
    in_channels=3,
    depth=10
).to(device)

# %%
# Create neural network functions (layers)
phi = torch.nn.Linear(1, 2).to(device)
weights_energy = torch.nn.Linear(num_classes, 1).to(device)
torch.nn.init.uniform_(weights_energy.weight)

criterion = VIRTUALOUTLIERSYNTHESIZER(phi, weights_energy, device=device,
                                      num_classes=num_classes,
                                      num_input_last_layer= 128, 
                                      fc = model.fc,
                                      sample_number=10,
                                      sample_from=20)

# %%
# Train model for some epochs
optimizer = torch.optim.Adam(list(model.parameters()) + list(phi.parameters()) +list(weights_energy.parameters())
                             , lr=lr)




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

ious = []
loss_ema = 0
ioe_ema = 0

for epoch in range(num_epochs):
    for n, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        y, x = y.to(device), x.to(device)

        y_hat = model(x)
        features=model.features(x)
        loss = criterion(y_hat,features ,y)

        loss.backward()
        optimizer.step()
        scheduler.step()

     

        loss_ema = 0.8 * loss_ema + 0.2 * loss.item()

        if n % 10 == 0:
            print(
                f"Epoch {epoch:03d} [{n:05d}/{len(loader):05d}] \t Loss: {loss_ema:02.2f}"
            )

# %%
# Evaluate
print("Evaluating")
model.eval()
loader= DataLoader(dataset + dataset_test, batch_size=batch_size, num_workers=12)
detector = WeightedEBO(model, weights_energy)
metrics = OODMetrics(mode="classification")

with torch.no_grad():
    for n, (x, y) in enumerate(loader):
        y, x = y.to(device), x.to(device)
        o = detector(x)
        
        metrics.update(o, y)
        if n % 10 == 0:
            print(
                f"Epoch {epoch:03d} [{n:05d}/{len(loader):05d}] "
            )

print(metrics.compute())

