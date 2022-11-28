import torch
import torchvision.transforms as tvt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.loss import CACLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, is_known

torch.manual_seed(123)

n_epochs = 10
device = "cuda:0"

trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor()])

# setup IN training data
dataset_in_train = CIFAR10(root="data", train=True, download=True, transform=trans)

# setup IN test data
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# setup OOD test data, use ToUnknown() to mark labels as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# create data loaders
train_loader = DataLoader(dataset_in_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=64)

# Create DNN, pretrained on the imagenet excluding cifar10 classes
model = WideResNet(num_classes=1000, pretrained="imagenet32-nocifar")
# we have to replace the final layer to match the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, 10)

model.to(device)

opti = Adam(model.parameters())
criterion = CACLoss(n_classes=10, magnitude=5, alpha=2).to(device)


def test():
    """
    Evaluate model
    """
    metrics = OODMetrics()
    acc = Accuracy(num_classes=10)

    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            # calculate embeddings
            z = model(x.to(device))
            # calculate the distance of each embedding to each center
            distances = criterion.distance(z).cpu()
            # the CAC Loss proposes its own method for score calculation.
            # We could, however, also use the minimum distance.
            metrics.update(CACLoss.score(distances), y)
            known = is_known(y)
            if known.any():
                acc.update(distances[known].min(dim=1).indices, y[known])

    print(metrics.compute())
    print(f"Accuracy: {acc.compute().item()}")
    model.train()


# start training
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    for x, y in train_loader:
        # calculate embeddings
        z = model(x.to(device))
        # calculate the distance of each embedding to each center
        distances = criterion.distance(z)
        # calculate CAC loss, based on distances to centers
        loss = criterion(distances, y.cuda())
        opti.zero_grad()
        loss.backward()
        opti.step()

    test()
