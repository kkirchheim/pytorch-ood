import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown

torch.manual_seed(123)
device = "cuda:0"

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor(), tvt.Normalize(std=std, mean=mean)])

# setup data
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# Stage 1: Create DNN
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device)

# Stage 2: Create Detector
detector = MaxSoftmax(model)

# Stage 3: Evaluate Detectors
metrics = OODMetrics()

for n, batch in enumerate(test_loader):
    x, y = batch
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
