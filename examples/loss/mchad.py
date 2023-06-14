"""
MCHAD
-------------------------

:class:`Multi Class Hypersphere Anomaly Detection <pytorch_ood.loss.MCHADLoss>` (MCHAD) can be seen as a
multi-class generalization of Deep One-Class Learning, where there are
several centers :math:`\\{\\mu_1, \\mu_2, ..., \\mu_y\\}` in the output space of the model, one for each class.
During training, the representation :math:`f_{\\theta}(x)` from class :math:`y` is drawn
towards the corresponding center :math:`\\mu_y`.

In contrast to Class Anchor Clustering, the position of the class-centers can be learned.
Also, the method allows to incorporate outliers into the training. On the downside, MCHAD requires more
hyperparameters.

Here, we train the model for 10 epochs on the CIFAR10 dataset, using a backbone pre-trained on the
:math:`32 \\times 32` resized version of the ImageNet as a foundation.
We use the TinyImages300k as training outlier data.

"""
import torch
import torchvision.transforms as tvt
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from pytorch_ood.dataset.img import Textures, TinyImages300k
from pytorch_ood.loss import MCHADLoss
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, is_known

fix_random_seed(123)

n_epochs = 10
device = "cuda:0"
embedding_dim = 5

trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor()])

# setup IN training data
data_in_train = CIFAR10(root="data", train=True, download=True, transform=trans)

# setup OOD training data, same site as IN training data
tiny300k = TinyImages300k(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)
data_out_train, _ = random_split(
    tiny300k, [len(data_in_train), len(tiny300k) - len(data_in_train)]
)

# setup IN test data
data_in_test = CIFAR10(root="data", train=False, transform=trans)

# setup OOD test data, use ToUnknown() to mark labels as OOD
data_out_test = Textures(root="data", download=True, transform=trans, target_transform=ToUnknown())

# create data loaders
train_loader = DataLoader(data_in_train + data_out_train, batch_size=64, shuffle=True)
test_loader = DataLoader(data_in_test + data_out_test, batch_size=64)

# %%
# Create DNN, pretrained on the imagenet excluding cifar10 classes.
# We have to replace the final layer to match the number of classes.
model = WideResNet(num_classes=1000, pretrained="imagenet32-nocifar")
model.fc = torch.nn.Linear(model.fc.in_features, embedding_dim)
model.to(device)

opti = Adam(model.parameters())
criterion = MCHADLoss(n_classes=10, n_dim=embedding_dim, weight_oe=0.001).to(device)

# %%
# Define a function that evaluates the model


def test():
    metrics = OODMetrics()
    acc = Accuracy(num_classes=10)

    model.eval()

    with torch.no_grad(), tqdm(test_loader, desc="Testing") as bar:
        for x, y in bar:
            # calculate embeddings
            z = model(x.to(device))
            # calculate the distance of each embedding to each center
            distances = criterion.distance(z).cpu()
            metrics.update(distances.min(dim=1).values, y)
            known = is_known(y)
            if known.any():
                acc.update(distances[known].min(dim=1).indices, y[known])

    print(metrics.compute())
    print(f"Accuracy: {acc.compute().item():.2%}")
    model.train()


# %%
# Start training

for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    loss_ema = 0

    with tqdm(train_loader, desc=f"Epoch {epoch}") as bar:
        for x, y in bar:
            # calculate embeddings
            z = model(x.to(device))
            # calculate the distance of each embedding to each center
            distances = criterion.distance(z)
            # calculate MCHAD loss, based on distances to centers
            loss = criterion(distances, y.cuda())
            opti.zero_grad()
            loss.backward()
            opti.step()

            loss_ema = 0.8 * loss_ema + 0.2 * loss.item()
            bar.set_postfix_str(f"loss: {loss.item()}")

    test()
