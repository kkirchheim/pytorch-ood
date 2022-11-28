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

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

trans = tvt.Compose([tvt.Resize(size=(32, 32)), tvt.ToTensor(), tvt.Normalize(std=std, mean=mean)])

# setup IN test data
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
# setup OOD test data
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

# merge dataset and create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# Stage 1: Create DNN
model = WideResNet(num_classes=10, pretrained="cifar10-pixmix", widen_factor=4).to(device).eval()

# Stage 2: Create Detector
detector = MaxSoftmax(model)

# Stage 3: Evaluate Detectors
metrics = OODMetrics()

for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
