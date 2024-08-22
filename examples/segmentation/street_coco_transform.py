"""
StreetHazards
-------------------------

We train a Feature Pyramid Segmentation model
with a ResNet-50 backbone pre-trained on the ImageNet
on the :class:`StreetHazards<pytorch_ood.dataset.img.StreetHazards>`.
We then use the :class:`EnergyBased<pytorch_ood.detector.EnergyBased>` OOD detector.

.. note :: Training with a batch-size of 4 requires slightly more than 12 GB of GPU memory.
    However, the models tend to also converge to reasonable performance with a smaller batch-size.

.. warning :: The results produced by this script vary. It is impossible to ensure the
    reproducibility of the exact numerical values at the moment, because the model includes operations for
    which no deterministic implementation exists at the time of writing.

.. note :: The license of the model originally used for the street hazards dataset
    is not compatible with ``pytorch-ood``. This prevents us from re-using the implementation
    from the original repository.

"""
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.metrics import iou_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad, to_tensor

from pytorch_ood.dataset.img import StreetHazards
from pytorch_ood.detector import Entropy
from pytorch_ood.loss import EntropicOpenSetLoss
from pytorch_ood.utils import OODMetrics, fix_random_seed
from pytorch_ood.utils.transforms import InsertCOCO

device = "cuda:0"
batch_size = 4
num_epochs = 1

fix_random_seed(12345)
g = torch.Generator()
g.manual_seed(0)


# %%
# Setup preprocessing
preprocess_input = get_preprocessing_fn("resnet50", pretrained="imagenet")

# for demonstration purposes, we set the probability of OOD to 1
coco_transform = InsertCOCO(
    coco_dir="data/coco",
    prohibet_classes="Streethazards",
    probability_of_ood=1,  # 0.1
    ood_per_image=1,
    ood_mask_value=-1,
    upscale=1.4150357439499515,
    year=2017,
    min_size_of_img=480,
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


# %%
# Setup datasets
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
    classes=13,
).to(device)

# %%
# Train model for some epochs
criterion = EntropicOpenSetLoss()
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
detector = Entropy(model)
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
# {'AUROC': 0.9573410749435425, 'AUPR-IN': 0.5151191353797913, 'AUPR-OUT': 0.9991346001625061, 'FPR95TPR': 0.16139476001262665}
