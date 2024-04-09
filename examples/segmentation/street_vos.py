"""
StreetHazards with VOS Loss
-------------------------------------

We train a Feature Pyramid Segmentation model
with a ResNet-50 backbone pre-trained on the ImageNet
on the :class:`StreetHazards<pytorch_ood.dataset.img.StreetHazards>` **test set** using
the supervised :class:`VOSRegLoss<pytorch_ood.loss.VOSRegLoss>`.

We then use the :class:`VOSBased<pytorch_ood.detector.VOSBased>` OOD detector.

This setup is merely made to demonstrate how to train a supervised anomaly segmentation model with
this loss function.

.. warning :: We train on the test set, as it contains examples of anomalies.
    The results will not be meaningful.


.. note :: Training with a batch-size of 4 requires slightly more than 12 GB of GPU memory.
    However, the models tend to also converge to reasonable performance with a smaller batch-size.
    This loss is more effektive with a scheduler and a lot of epochs.

"""
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.metrics import iou_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad, to_tensor

from pytorch_ood.dataset.img import StreetHazards
from pytorch_ood.detector import VOSBased
from pytorch_ood.loss import VOSRegLoss
from pytorch_ood.utils import OODMetrics, fix_random_seed

device = "cuda:0"
batch_size = 4
num_epochs = 1
lr = 0.0001
num_classes = 13

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

    # size must be divisible by 32, so we pad the image.
    img = pad(img, [0, 8]).float()
    target = pad(target, [0, 8])
    return img, target


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# %%
# Setup datasets, train on ood images for demonstration purposes.
dataset = StreetHazards(root="data", subset="test", transform=my_transform, download=True)
dataset_test = StreetHazards(root="data", subset="test", transform=my_transform, download=True)


# %%
# Setup model
model = smp.FPN(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
).to(device)

# %%
# Create non-linear neural network functions
phi = torch.nn.Linear(1, 2).to(device)
weights_energy = torch.nn.Linear(num_classes, 1).to(device)
torch.nn.init.uniform_(weights_energy.weight)

criterion = VOSRegLoss(phi, weights_energy, device=device)


# %%
# Train model for some epochs
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    worker_init_fn=fix_random_seed,
    generator=g,
)

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
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        tp, fp, fn, tn = smp.metrics.get_stats(
            y_hat.softmax(dim=1).max(dim=1).indices.long(),
            y.long(),
            mode="multiclass",
            num_classes=13,
        )
        iou = iou_score(tp, fp, fn, tn)

        loss_ema = 0.8 * loss_ema + 0.2 * loss.item()
        ioe_ema = 0.8 * ioe_ema + 0.2 * iou.mean().item()

        if n % 10 == 0:
            print(
                f"Epoch {epoch:03d} [{n:05d}/{len(loader):05d}] \t Loss: {loss_ema:02.2f} \t IoU: {ioe_ema:02.2f}"
            )

# %%
# Evaluate
print("Evaluating")
model.eval()
loader = DataLoader(dataset_test, batch_size=4, worker_init_fn=fix_random_seed, generator=g)
detector = VOSBased(model, weights_energy)
metrics = OODMetrics(mode="segmentation")

with torch.no_grad():
    for n, (x, y) in enumerate(loader):
        y, x = y.to(device), x.to(device)
        o = detector(x)

        # undo padding
        o = pad(o, [-8, -8])
        y = pad(y, [-8, -8])

        metrics.update(o, y)

print(metrics.compute())


# %%
# Output:
# {'AUROC': 0.9346237778663635, 'AUPR-IN': 0.15255042910575867, 'AUPR-OUT': 0.9993401169776917, 'FPR95TPR': 0.18086743354797363}
