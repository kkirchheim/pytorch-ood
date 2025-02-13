"""
RoadAnomaly
-------------------------

We train a Feature Pyramid Segmentation model
with a ResNet-50 backbone pre-trained on the ImageNet
on the `Citiscapes <https://www.cityscapes-dataset.com/>`__ Dataset (please download  it before and put **gtFine** and **leftImg8bit** it into the **data/cityscapes** folder).
This model is evaluated using the :class:`EnergyBased<pytorch_ood.detector.EnergyBased>` OOD detector on the original :class:`RoadAnomaly<pytorch_ood.dataset.img.RoadAnomaly>` dataset and both datasets of the :class:`SegmentMeIfYouCan<pytorch_ood.dataset.img.SegmentMeIfYouCan>` benchmark: RoadAnomaly21 and RoadObstacles21.


.. note :: Training with a batch-size of 4 requires slightly more than 12 GB of GPU memory.
    However, the models tend to also converge to reasonable performance with a smaller batch-size.

.. warning :: The results produced by this script vary. It is impossible to ensure the
    reproducibility of the exact numerical values at the moment, because the model includes operations for
    which no deterministic implementation exists at the time of writing.

"""

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.metrics import iou_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad, to_tensor
from torchvision.datasets import Cityscapes
from PIL import Image

from pytorch_ood.dataset.img import RoadAnomaly, SegmentMeIfYouCan
from pytorch_ood.detector import EnergyBased
from pytorch_ood.utils import OODMetrics, fix_random_seed

device = "cuda:0"
batch_size = 4
num_epochs = 1
classes = 34
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

    # case image is not 1280x720
    H, W = img.shape[-2:]
    if H != 720 or W != 1280:
        img = torch.nn.functional.interpolate(
            img[None, ...], size=(720, 1280), mode="bilinear", align_corners=False
        )[0]
        target = torch.nn.functional.interpolate(
            target[None, None, ...].float(), size=(720, 1280), mode="nearest"
        )[0, 0].long()

    # size must be divisible by 32, so we pad the image.
    img = pad(img, [0, 8]).float()
    target = pad(target, [0, 8])
    return img, target


def cityscapes_transform(img, target):
    # resize image and target to 1280,720
    img = img.resize((1280, 720))
    # use nearest neighbour interpolation for target
    target = target.resize((1280, 720), Image.NEAREST)
    target = to_tensor(target).squeeze(0)
    target = target = (target * 255).long()
    return my_transform(img, target)


def eval(dataset_test, detector):
    metrics = OODMetrics(mode="segmentation", void_label=1)
    loader = DataLoader(dataset_test, batch_size=4, worker_init_fn=fix_random_seed, generator=g)

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
# Setup datasets

# Please download Citiscapes Dataset, for example, from https://www.cityscapes-dataset.com/ or https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/download/downloader.py
# and put it in the data/cityscapes folder
dataset = Cityscapes(
    root="data/cityscapes",
    split="train",
    transforms=cityscapes_transform,
    mode="fine",
    target_type="semantic",
)

# Test datasets for RoadAnomaly
dataset_test_roadanomaly_original = RoadAnomaly(root="data", transform=my_transform, download=True)

# Test datasets for SegmentMeIfYouCan
dataset_test_SMIYC_RoadAnomaly21 = SegmentMeIfYouCan(
    root="data", subset="RoadAnomaly21", transform=my_transform, download=True
)
dataset_test_SMIYC_RoadObstacle21 = SegmentMeIfYouCan(
    root="data", subset="RoadObstacle21", transform=my_transform, download=True
)


# %%
# Setup model
model = smp.FPN(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=classes,
).to(device)

# %%
# Train model for some epochs
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    worker_init_fn=fix_random_seed,
    generator=g,
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

        tp, fp, fn, tn = smp.metrics.get_stats(
            y_hat.softmax(dim=1).max(dim=1).indices.long(),
            y.long(),
            mode="multiclass",
            num_classes=classes,
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
detector = EnergyBased(model)

print("RoadAnomaly Original dataset")
eval(dataset_test_roadanomaly_original, detector)
print("SegmentMeIfYouCan RoadAnomaly21 dataset")
eval(dataset_test_SMIYC_RoadAnomaly21, detector)
print("SegmentMeIfYouCan RoadObstacle21 dataset")
eval(dataset_test_SMIYC_RoadObstacle21, detector)


# %%
# Output:
#
# +----------------------------------+-----------+--------+--------+---------+----------+----------+
# | Dataset                          | Detector  | AUROC  | AUTC   | AUPR-IN | AUPR-OUT | FPR95TPR |
# +==================================+===========+========+========+=========+==========+==========+
# | RoadAnomaly Original             | Energy    | 80.90  | 41.66  | 97.00   | 29.76    | 46.31    |
# +----------------------------------+-----------+--------+--------+---------+----------+----------+
# | SegmentMeIfYouCan RoadAnomaly21  | Energy    | 83.69  | 40.69  | 96.31   | 43.54    | 47.92    |
# +----------------------------------+-----------+--------+--------+---------+----------+----------+
# | SegmentMeIfYouCan RoadObstacle21 | Energy    | 87.34  | 35.00  | 99.97   | 28.34    | 20.45    |
# +----------------------------------+-----------+--------+--------+---------+----------+----------+
