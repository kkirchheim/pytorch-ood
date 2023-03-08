"""


CIFAR 100
==============================
The evaluation is the same as for CIFAR 10.

+-------------+-------+---------+----------+----------+
| Detector    | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=========+==========+==========+
| MaxSoftmax  | 79.10 | 73.75   | 82.92    | 58.16    |
+-------------+-------+---------+----------+----------+
| KLMatching  | 80.14 | 75.26   | 82.48    | 60.73    |
+-------------+-------+---------+----------+----------+
| ODIN        | 81.46 | 76.67   | 84.88    | 55.55    |
+-------------+-------+---------+----------+----------+
| Mahalanobis | 83.91 | 79.86   | 86.08    | 47.03    |
+-------------+-------+---------+----------+----------+
| MaxLogit    | 84.64 | 79.95   | 87.11    | 48.51    |
+-------------+-------+---------+----------+----------+
| EnergyBased | 84.90 | 80.27   | 87.32    | 47.84    |
+-------------+-------+---------+----------+----------+
| ViM         | 85.86 | 81.18   | 88.81    | 41.82    |
+-------------+-------+---------+----------+----------+




"""
import pandas as pd  # additional dependency, used here for convenience
import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
)
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToRGB, ToUnknown

device = "cuda:0"

torch.manual_seed(123)

# setup preprocessing
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
trans = tvt.Compose(
    [tvt.Resize(size=(32, 32)), ToRGB(), tvt.ToTensor(), tvt.Normalize(std=std, mean=mean)]
)

# %%
# Setup datasets
dataset_in_test = CIFAR100(root="data", train=False, transform=trans, download=True)

# create all OOD datasets
ood_datasets = [Textures, TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=256)
    datasets[ood_dataset.__name__] = test_loader

# %%
# **Stage 1**: Create DNN with pre-trained weights from the Hendrycks baseline paper
print("STAGE 1: Creating a Model")
model = WideResNet(num_classes=100, pretrained="cifar100-pt").eval().to(device)

# Stage 2: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis"] = Mahalanobis(model.features, norm_std=std, eps=0.002)
detectors["KLMatching"] = KLMatching(model)
detectors["MaxSoftmax"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)
detectors["ODIN"] = ODIN(model, norm_std=std, eps=0.002)

# %%
# **Stage 2**: fit detectors to training data (some require this, some do not)
print(f"> Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(CIFAR100(root="data", train=True, transform=trans), batch_size=256)
for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.fit(loader_in_train, device=device)

# %%
# **Stage 3**: Evaluate Detectors
print(f"STAGE 3: Evaluating {len(detectors)} detectors on {len(datasets)} datasets.")
results = []

with torch.no_grad():
    for detector_name, detector in detectors.items():
        print(f"> Evaluating {detector_name}")
        for dataset_name, loader in datasets.items():
            print(f"--> {dataset_name}")
            metrics = OODMetrics()
            for x, y in loader:
                metrics.update(detector(x.to(device)), y.to(device))

            r = {"Detector": detector_name, "Dataset": dataset_name}
            r.update(metrics.compute())
            results.append(r)

# calculate mean scores over all datasets, use percent
df = pd.DataFrame(results)
mean_scores = df.groupby("Detector").mean() * 100
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
