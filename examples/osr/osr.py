"""
CIFAR10
-------------------------

Open Set Simulation on CIFAR 10

"""

import torch.nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchmetrics import Accuracy

from pytorch_ood.dataset.ossim import DynamicOSS
from pytorch_ood.model import WideResNet
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import fix_random_seed, TargetMapping, OODMetrics, is_known

device = "cuda:0"
num_epochs = 10

fix_random_seed(12345)

# %%
# Setup preprocessing
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# %%
# Setup datasets
dataset_1 = CIFAR10(root="data", train=True, transform=trans, download=True)
dataset_2 = CIFAR10(root="data", train=False, transform=trans, download=True)
dataset = dataset_1 + dataset_2

# %%
# Create DNN with pre-trained on a downscaled version of the image net, excluding cifar images
# adjust it to output 7 logits
print("Creating a Model")
model = WideResNet(num_classes=1000, pretrained="imagenet32-nocifar")
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.to(device)

# %%
# Create open set simulation and dataloaders
# We use 3 classes as unknown unknown (OOD), and a data split of 90% train and 10% test
ossim = DynamicOSS(
    dataset=dataset,
    train_size=0.9,
    val_size=0.0,
    test_size=0.1,
    kuc=0,
    uuc_val=0,
    uuc_test=3,
    seed=1,
)
print(f"Known Classes: {ossim.kkc}")
print(f"Unknown Classes: {ossim.uuc}")

# create class remapping
class_mapping = TargetMapping(known=ossim.kkc, unknown=ossim.uuc)

train_loader = DataLoader(ossim.train_dataset(), batch_size=32, num_workers=12)
test_loader = DataLoader(ossim.test_dataset(), batch_size=32, num_workers=12)

criterion = CrossEntropyLoss()

opti = torch.optim.Adam(model.parameters(), lr=0.001)


# %%
# Define function for testing
@torch.no_grad()
def test():
    metrics = OODMetrics()
    acc = Accuracy(task="multiclass", num_classes=7).to(device)
    model.eval()

    for x, y in tqdm(test_loader):
        # do not forget to remap class labels
        y = torch.tensor([class_mapping(i.item()) for i in y])

        y = y.to(device)
        x = x.to(device)

        z = model(x)

        metrics.update(MaxSoftmax.score(z), y)

        known = is_known(y)
        if known.any():
            acc.update(z[known].argmax(dim=1), y[known])

    print(metrics.compute())
    print(acc.compute().item())


# %%
# Start training
for epoch in range(num_epochs):
    bar = tqdm(train_loader)
    model.train()
    loss_ema = None
    for x, y in bar:
        # do not forget to remap class labels
        y = torch.tensor([class_mapping(i.item()) for i in y])

        y = y.to(device)
        x = x.to(device)

        z = model(x)

        loss = criterion(z, y)
        opti.zero_grad()
        loss.backward()
        opti.step()

        loss_ema = loss_ema * 0.95 + loss.item() * 0.05 if loss_ema is not None else loss.item()
        bar.set_postfix_str(f"loss: {loss_ema:.2f}")

    test()
