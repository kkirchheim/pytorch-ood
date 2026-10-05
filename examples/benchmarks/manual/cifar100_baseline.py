"""

CIFAR 100
==============================

The evaluation is the same as for CIFAR 10.

+------------------+-------+-------+---------+----------+----------+
| Detector         | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
+==================+=======+=======+=========+==========+==========+
| ODIN             | 86.39 | 41.38 | 79.86   | 88.92    | 43.74    |
+------------------+-------+-------+---------+----------+----------+
| Gram             | 85.56 | 49.80 | 75.80   | 91.36    | 46.89    |
+------------------+-------+-------+---------+----------+----------+
| MultiMahalanobis | 85.34 | 45.92 | 77.85   | 89.50    | 39.26    |
+------------------+-------+-------+---------+----------+----------+
| fDBD             | 85.23 | 37.83 | 77.54   | 88.08    | 45.25    |
+------------------+-------+-------+---------+----------+----------+
| GEN              | 84.87 | 34.51 | 78.38   | 86.67    | 46.41    |
+------------------+-------+-------+---------+----------+----------+
| EnergyBased      | 84.72 | 41.94 | 77.97   | 86.58    | 47.09    |
+------------------+-------+-------+---------+----------+----------+
| DICE             | 84.57 | 42.83 | 78.33   | 86.57    | 44.42    |
+------------------+-------+-------+---------+----------+----------+
| MaxLogit         | 84.43 | 41.95 | 77.61   | 86.37    | 47.79    |
+------------------+-------+-------+---------+----------+----------+
| RMD              | 82.03 | 39.65 | 73.77   | 85.52    | 51.58    |
+------------------+-------+-------+---------+----------+----------+
| ViM              | 81.63 | 43.53 | 72.79   | 85.79    | 50.00    |
+------------------+-------+-------+---------+----------+----------+
| Entropy          | 80.97 | 38.48 | 72.88   | 84.36    | 56.76    |
+------------------+-------+-------+---------+----------+----------+
| KLMatching       | 79.41 | 39.29 | 67.65   | 83.13    | 62.19    |
+------------------+-------+-------+---------+----------+----------+
| Mahalanobis+ODIN | 79.25 | 44.78 | 68.70   | 84.59    | 55.80    |
+------------------+-------+-------+---------+----------+----------+
| SHE              | 79.03 | 43.74 | 72.98   | 81.89    | 50.53    |
+------------------+-------+-------+---------+----------+----------+
| MSP              | 78.56 | 37.48 | 71.15   | 82.14    | 57.90    |
+------------------+-------+-------+---------+----------+----------+
| Mahalanobis      | 73.46 | 45.87 | 64.19   | 80.34    | 60.06    |
+------------------+-------+-------+---------+----------+----------+
| GMM              | 72.40 | 46.64 | 62.77   | 79.57    | 62.79    |
+------------------+-------+-------+---------+----------+----------+
| RankFeat         | 62.35 | 49.07 | 51.88   | 70.28    | 80.93    |
+------------------+-------+-------+---------+----------+----------+


"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/benchmarks.png"

import pandas as pd  # additional dependency, used here for convenience
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Places365,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
)
from pytorch_ood.detector import (
    DICE,
    GEN,
    GMM,
    ODIN,
    RMD,
    SHE,
    EnergyBased,
    Entropy,
    Gram,
    KLMatching,
    Mahalanobis,
    MahalanobisODIN,
    MaxLogit,
    MaxSoftmax,
    MultiMahalanobis,
    RankFeat,
    ViM,
    fDBD,
)
from pytorch_ood.metrics import OODMetrics
from pytorch_ood.model import get_model_info, load_model, load_transform
from pytorch_ood.utils import ToUnknown, fix_random_seed

device = "cuda:0"

fix_random_seed(123)

# setup preprocessing
trans = load_transform("wrn-40-2/cifar100/crossentropy")
norm_std = get_model_info("wrn-40-2/cifar100/crossentropy").preprocessing.std

# %%
# Setup datasets
dataset_in_test = CIFAR100(root="data", train=False, transform=trans, download=True)

# create all OOD datasets
ood_datasets = [
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    LSUNCrop,
    LSUNResize,
    Places365,
    CIFAR10,
    MNIST,
    FashionMNIST,
]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=256, num_workers=12)
    datasets[ood_dataset.__name__] = test_loader

# %%
# **Stage 1**: Create DNN with pre-trained weights from the Hendrycks baseline paper
print("STAGE 1: Creating a Model")
model = load_model("wrn-40-2/cifar100/crossentropy").to(device)

# Stage 2: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}
detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis+ODIN"] = MahalanobisODIN(model.features, norm_std=norm_std, eps=0.002)
detectors["Mahalanobis"] = Mahalanobis(model.features)
detectors["KLMatching"] = KLMatching(model)
detectors["SHE"] = SHE(model.features, model.fc)
detectors["MSP"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["GEN"] = GEN(model)
detectors["GMM"] = GMM(model.features)
detectors["fDBD"] = fDBD(encoder=model.features, head=model.fc)
detectors["RankFeat"] = RankFeat(backbone=model.feature_maps, head=model.forward_feature_maps)
detectors["MaxLogit"] = MaxLogit(model)
detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)
detectors["DICE"] = DICE(encoder=model.features, w=model.fc.weight, b=model.fc.bias, p=0.65)
detectors["RMD"] = RMD(model.features)
detectors["MultiMahalanobis"] = MultiMahalanobis(
    [
        model.conv1,
        model.block1,
        model.block2,
        model.block3,
        nn.Sequential(model.bn1, model.relu),
    ]
)
detectors["Gram"] = Gram(
    num_classes=100,
    head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc),
    feature_layers=[
        model.conv1,
        model.block1,
        model.block2,
        model.block3,
        nn.Sequential(model.bn1, model.relu),
    ],
)

# %%
# **Stage 2**: fit detectors to training data (some require this, some do not)
print(f"> Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(
    CIFAR100(root="data", train=True, transform=trans), batch_size=256, num_workers=12
)
for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.to(device)
    detector.fit(loader_in_train)

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
mean_scores = (
    df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
)
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
