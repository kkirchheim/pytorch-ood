import torch
import torchvision.transforms as tvt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import AUROC  # additional dependency
from torchvision.datasets import CIFAR10

from oodtk import NegativeEnergy, Softmax
from oodtk.dataset.img import Textures
from oodtk.model import WideResNet
from oodtk.utils import is_known

torch.manual_seed(123)

trans = tvt.Compose([tvt.Resize(32), tvt.ToTensor()])

# setup data
dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)
dataset_out_test = Textures(root="data", download=True, transform=trans)
dataset_test = ConcatDataset([dataset_in_test, dataset_out_test])
train_loader = DataLoader(dataset_train, batch_size=128)
test_loader = DataLoader(dataset_test)

# setup model
model = WideResNet(depth=40, num_classes=10, widen_factor=2, drop_rate=0.3)
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


# create some methods
energy = NegativeEnergy()
softmax = Softmax()

# evaluate
auroc_energy = AUROC(num_classes=2)
auroc_softmax = AUROC(num_classes=2)
model.eval()

for n, batch in enumerate(test_loader):
    x, y = batch
    logits = model(x)
    y_hat = logits.argmax(dim=1)

    auroc_energy.update(energy(logits), is_known(y))
    auroc_softmax.update(softmax(logits), is_known(y))

print(auroc_softmax.compute())
print(auroc_energy.compute())
