"""
StreetHazards with VOS Loss
-------------------------------------

We train a Feature Pyramid Segmentation model
with a ResNet-50 backbone pre-trained on the ImageNet
on the :class:`StreetHazards<pytorch_ood.dataset.img.StreetHazards>` training set using
the supervised :class:`VOSRegLoss<pytorch_ood.loss.VOSRegLoss>`.
The loss needs anomalous pixels, which the training set does not contain, so we insert random
COCO objects as anomalies with :class:`InsertCOCO<pytorch_ood.augment.img.InsertCOCO>`.

We then use the :class:`WeightedEBO<pytorch_ood.detector.WeightedEBO>` OOD detector.

This setup is merely made to demonstrate how to train a supervised anomaly segmentation model with
this loss function.

.. note :: Training with a batch-size of 4 requires slightly more than 12 GB of GPU memory.
    However, the models tend to also converge to reasonable performance with a smaller batch-size.
    This loss is more effektive with a scheduler and a lot of epochs.

"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/segmentation.png"

import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.metrics import iou_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad, to_tensor

from pytorch_ood.augment.img import InsertCOCO
from pytorch_ood.dataset.img import StreetHazards
from pytorch_ood.detector import WeightedEBO
from pytorch_ood.loss import VOSRegLoss
from pytorch_ood.metrics import OODPerImageSegmentationMetrics
from pytorch_ood.utils import fix_random_seed

device = "cuda:0"
batch_size = 1
num_epochs = 1
lr = 0.0001
num_classes = 13

fix_random_seed(12345)
g = torch.Generator()
g.manual_seed(0)


# %%
# Setup preprocessing
preprocess_input = get_preprocessing_fn("resnet50", pretrained="imagenet")

# for demonstration purposes, we set the probability of OOD to 1
coco_transform = InsertCOCO(
    coco_dir="data/coco",
    exclude_classes="Streethazards",
    p=1,
    download=True,
)


def my_transform(img, target, use_coco_transform):
    if use_coco_transform:
        img, target = coco_transform(img, target)
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
# Setup datasets, insert COCO objects into the training images only.
dataset = StreetHazards(
    root="data",
    subset="train",
    transform=lambda img, target: my_transform(img, target, True),
    download=True,
)
dataset_test = StreetHazards(
    root="data",
    subset="test",
    transform=lambda img, target: my_transform(img, target, False),
    download=True,
)


# %%
# Setup model
model = smp.FPN(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
).to(device)

# %%
# Create neural network functions (layers)
phi = torch.nn.Linear(1, 2).to(device)
weights_energy = torch.nn.Linear(num_classes, 1).to(device)
torch.nn.init.uniform_(weights_energy.weight)

criterion = VOSRegLoss(phi, weights_energy).to(device)


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
detector = WeightedEBO(model, weights_energy.weight)
# mean over the images
metrics = OODPerImageSegmentationMetrics()

with torch.no_grad():
    for n, (x, y) in enumerate(loader):
        y, x = y.to(device), x.to(device)
        o = detector(x)

        # undo padding
        o = pad(o, [0, -8])
        y = pad(y, [0, -8])

        metrics.update(o, y)

print(metrics.compute())


# %%
# Output:
#
# +--------------------+-------------+-------+-------+---------+----------+----------+
# | Dataset            | Detector    | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
# +====================+=============+=======+=======+=========+==========+==========+
# | StreetHazards+COCO | WeightedEBO | 90.95 | 38.93 | 99.87   | 11.68    | 28.51    |
# +--------------------+-------------+-------+-------+---------+----------+----------+
