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

# setup ID training data
dataset_in_train = CIFAR10(root="data", train=True, download=True, transform=trans)

# setup OOD training data, use ToUnknown() to mark labels as OOD
# this way, outlier exposure can automatically decide if the training samples are ID or OOD
dataset_out_train = TinyImages300k(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# setup ID test data
dataset_in_test = CIFAR10(root="data", train=False, transform=trans)

# setup OOD test data, use ToUnknown() to mark labels as OOD
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# create data loaders
train_loader = DataLoader(dataset_in_train + dataset_out_train, batch_size=64, shuffle=True)
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

# %%
# Output:
# Epoch 0
# {'AUROC': 0.9438387155532837, 'AUPR-IN': 0.9145375490188599, 'AUPR-OUT': 0.9601001143455505, 'FPR95TPR': 0.3043999969959259}
# Epoch 1
# {'AUROC': 0.9723063111305237, 'AUPR-IN': 0.9310603737831116, 'AUPR-OUT': 0.9854252338409424, 'FPR95TPR': 0.10440000146627426}
# Epoch 2
# {'AUROC': 0.9726285338401794, 'AUPR-IN': 0.9353838562965393, 'AUPR-OUT': 0.9854604005813599, 'FPR95TPR': 0.10670000314712524}
# Epoch 3
# {'AUROC': 0.9664252996444702, 'AUPR-IN': 0.9456377625465393, 'AUPR-OUT': 0.9795949459075928, 'FPR95TPR': 0.18469999730587006}
# Epoch 4
# {'AUROC': 0.9807416200637817, 'AUPR-IN': 0.9635397791862488, 'AUPR-OUT': 0.9889929294586182, 'FPR95TPR': 0.09440000355243683}
# Epoch 5
# {'AUROC': 0.9845513701438904, 'AUPR-IN': 0.9637761116027832, 'AUPR-OUT': 0.9917054772377014, 'FPR95TPR': 0.06310000270605087}
# Epoch 6
# {'AUROC': 0.9830336570739746, 'AUPR-IN': 0.9557808637619019, 'AUPR-OUT': 0.9912922382354736, 'FPR95TPR': 0.05920000001788139}
# Epoch 7
# {'AUROC': 0.9907971620559692, 'AUPR-IN': 0.9819102883338928, 'AUPR-OUT': 0.9949544668197632, 'FPR95TPR': 0.04010000079870224}
# Epoch 8
# {'AUROC': 0.9874339699745178, 'AUPR-IN': 0.9670255184173584, 'AUPR-OUT': 0.9937484860420227, 'FPR95TPR': 0.041200000792741776}
# Epoch 9
# {'AUROC': 0.9871684312820435, 'AUPR-IN': 0.9701670408248901, 'AUPR-OUT': 0.9931120276451111, 'FPR95TPR': 0.051899999380111694}
