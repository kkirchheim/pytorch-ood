"""

Outlier Exposure
-------------------------

We train a model with  :class:`Outlier Exposure <pytorch_ood.loss.OutlierExposureLoss>` on the CIFAR10.

We can use a model pre-trained on the :math:`32 \\times 32` resized version of the ImageNet as a foundation.
As outlier data, we use :class:`TinyImages300k <pytorch_ood.dataset.img.TinyImages300k>`, a cleaned version of the
TinyImages database, which contains random images scraped from the internet.


"""
import torch
import torchvision.transforms as tvt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures, TinyImages300k
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.loss import OutlierExposureLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown

torch.manual_seed(123)

# maximum number of epochs and training iterations
n_epochs = 10
device = "cuda:0"

# %%
# Setup preprocessing and data
trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor()])

# setup IN training data
dataset_in_train = CIFAR10(root="data", train=True, download=True, transform=trans)

# setup OOD training data, use ToUnknown() to mark labels as OOD
# this way, outlier exposure can automatically decide if the training samples are IN or OOD
dataset_out_train = TinyImages300k(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# setup IN test data
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# setup OOD test data, use ToUnknown() to mark labels as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# create data loaders
train_loader = DataLoader(
    dataset_in_train + dataset_out_train, batch_size=64, shuffle=True
)
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=64)

# %%
# Create DNN, pretrained on the imagenet excluding cifar10 classes
model = WideResNet(num_classes=1000, pretrained="imagenet32-nocifar")
# we have to replace the final layer to account for the lower number of
# classes in the CIFAR10 dataset
model.fc = torch.nn.Linear(model.fc.in_features, 10)

model.to(device)

opti = Adam(model.parameters())
criterion = OutlierExposureLoss(alpha=0.5)


# %%
# Define a function to test the model
def test():
    softmax = MaxSoftmax(model)

    metrics_softmax = OODMetrics()
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            metrics_softmax.update(softmax(x.to(device)), y)

    print(metrics_softmax.compute())
    model.train()


# %%
# Start training
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    for x, y in train_loader:
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))
        opti.zero_grad()
        loss.backward()
        opti.step()

    test()
