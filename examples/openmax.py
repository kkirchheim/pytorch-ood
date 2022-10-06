import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown

torch.manual_seed(123)
device = "cuda:0"

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor(), tvt.Normalize(std=std, mean=mean)])

# setup data
dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)

# create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# Stage 1: Create DNN
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()

# Stage 2: Create and Fit OpenMax
detector = OpenMax(model, tailsize=25, alpha=5, euclid_weight=0.5)
detector.fit(train_loader, device=device)

# Stage 3: Evaluate Detectors
metrics = OODMetrics()

for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
