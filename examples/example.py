import torch
import torchvision.transforms as tvt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood import NegativeEnergy, Softmax
from pytorch_ood.dataset.img import Textures
from pytorch_ood.metrics import OODMetrics
from pytorch_ood.model import WideResNet
from pytorch_ood.transforms import ToUnknown

torch.manual_seed(123)
max_iterations = 1

trans = tvt.Compose([tvt.Resize(32), tvt.ToTensor()])

# setup data
dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# treat samples from this dataset as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# concatenate datasets
dataset_test = dataset_in_test + dataset_out_test
train_loader = DataLoader(dataset_train, batch_size=64)
test_loader = DataLoader(dataset_test, batch_size=64)

# setup model
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


# create some methods
energy = NegativeEnergy(model)
softmax = Softmax(model)

# evaluate
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
