import torch
import torchvision.transforms as tvt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import EnergyBased, MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown

torch.manual_seed(123)
max_iterations = 1

trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor()])

# setup data
dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# treat samples from this dataset as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# create data loaders
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=64)

# Stage 1: Create DNN
model = WideResNet(num_classes=10)
opti = Adam(model.parameters())
criterion = CrossEntropyLoss()


# start training
for n, batch in enumerate(train_loader):
    x, y = batch
    logits = model(x)
    loss = criterion(logits, y)
    opti.zero_grad()
    loss.backward()
    opti.step()

    if n >= max_iterations:
        break


# Stage 2: Create OOD detector
# Fitting is not required in this case
energy = EnergyBased(model)
softmax = MaxSoftmax(model)

# Stage 3: Evaluate Detectors
metrics_energy = OODMetrics()
metrics_softmax = OODMetrics()
model.eval()

with torch.no_grad():
    for n, batch in enumerate(test_loader):
        x, y = batch
        metrics_energy.update(energy(x), y)
        metrics_softmax.update(softmax(x), y)


print(metrics_energy.compute())
print(metrics_softmax.compute())
