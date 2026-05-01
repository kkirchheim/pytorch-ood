"""

CIFAR 10
==============================

Example benchmark code for CIFAR10

+------------------+-------+-------+---------+----------+----------+
| Detector         | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
+==================+=======+=======+=========+==========+==========+
| GradNorm         | 50.00 | 60.78 | 18.37   | 81.63    | 100.00   |
+------------------+-------+-------+---------+----------+----------+
| Gram             | 69.37 | 46.01 | 58.02   | 77.49    | 75.03    |
+------------------+-------+-------+---------+----------+----------+
| KLMatching       | 88.48 | 39.83 | 72.29   | 91.33    | 57.84    |
+------------------+-------+-------+---------+----------+----------+
| NAC-UE           | 88.74 | 39.89 | 81.40   | 90.36    | 46.12    |
+------------------+-------+-------+---------+----------+----------+
| SHE              | 90.08 | 39.69 | 69.17   | 92.92    | 38.48    |
+------------------+-------+-------+---------+----------+----------+
| MSP              | 91.41 | 37.07 | 86.36   | 92.42    | 29.93    |
+------------------+-------+-------+---------+----------+----------+
| Entropy          | 92.03 | 35.90 | 86.70   | 93.47    | 29.75    |
+------------------+-------+-------+---------+----------+----------+
| Mahalanobis      | 92.14 | 42.76 | 86.39   | 94.36    | 28.25    |
+------------------+-------+-------+---------+----------+----------+
| ODIN             | 92.14 | 47.06 | 84.98   | 94.46    | 34.43    |
+------------------+-------+-------+---------+----------+----------+
| ViM              | 92.32 | 40.22 | 85.76   | 94.93    | 29.49    |
+------------------+-------+-------+---------+----------+----------+
| Mahalanobis+ODIN | 92.60 | 42.76 | 86.81   | 95.08    | 27.11    |
+------------------+-------+-------+---------+----------+----------+
| KNN              | 92.67 | 36.61 | 87.07   | 94.21    | 29.49    |
+------------------+-------+-------+---------+----------+----------+
| DICE             | 92.80 | 35.83 | 86.68   | 94.20    | 32.35    |
+------------------+-------+-------+---------+----------+----------+
| MultiMahalanobis | 92.89 | 45.35 | 85.51   | 96.09    | 24.95    |
+------------------+-------+-------+---------+----------+----------+
| MaxLogit         | 93.05 | 35.84 | 87.01   | 94.40    | 31.31    |
+------------------+-------+-------+---------+----------+----------+
| ASH              | 93.06 | 35.70 | 77.62   | 94.43    | 31.31    |
+------------------+-------+-------+---------+----------+----------+
| EnergyBased      | 93.11 | 35.45 | 87.09   | 94.46    | 31.14    |
+------------------+-------+-------+---------+----------+----------+
| RMD              | 93.46 | 32.09 | 87.73   | 95.08    | 26.99    |
+------------------+-------+-------+---------+----------+----------+



"""

import pandas as pd  # additional dependency, used here for convenience
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from copy import deepcopy
from tqdm.auto import tqdm  # additional dependency, used here for convenience
import torch

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    Places365,
)
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    GEN,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    RMD,
    DICE,
    SHE,
    Gram,
    GMM,
    MultiMahalanobis,
    NACUE,
    GradNorm,
    GradNormKL,
    ASH,
    KNN,
    RankFeat,
    fDBD,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

device = "cuda:0"

fix_random_seed(123)

# %%
# Setup preprocessing
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# %%
# Setup datasets

dataset_in_test = CIFAR10(root="data", train=False, transform=trans, download=True)

# create all OOD datasets
ood_datasets = [
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    LSUNCrop,
    LSUNResize,
    Places365,
    CIFAR100,
    MNIST,
    FashionMNIST,
]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128, num_workers=12)
    datasets[ood_dataset.__name__] = test_loader

# %%
# **Stage 1**: Create DNN with pre-trained weights from the Hendrycks baseline paper
print("STAGE 1: Creating a Model")
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)

# %%
# **Stage 2**: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}

detectors["KNN"] = KNN(model.features)
detectors["GMM"] = GMM(model.features)
detectors["fDBD"] = fDBD(encoder=model.features, head=model.fc)

detectors["ASH"] = ASH(backbone=model.feature_maps, head=model.forward_feature_maps)
detectors["RankFeat"] = RankFeat(backbone=model.feature_maps, head=model.forward_feature_maps)

# we make a copy of the model just so deactivating gradients does not influence other detectors
model_gn = deepcopy(model)
model_gn.requires_grad_(False)
model_gn.fc.requires_grad_(True)
detectors["GradNorm"] = GradNorm(model_gn, param_filter=lambda name: name.startswith("fc"))

model_gnkl = deepcopy(model)
model_gnkl.requires_grad_(False)
model_gnkl.fc.requires_grad_(True)
detectors["GradNormKL"] = GradNormKL(model_gnkl, param_filter=lambda name: name.startswith("fc"))

detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis+ODIN"] = Mahalanobis(model.features, norm_std=norm_std, eps=0.002)
detectors["Mahalanobis"] = Mahalanobis(model.features)

detectors["KLMatching"] = KLMatching(model)
detectors["SHE"] = SHE(model.features, model.fc)
detectors["MSP"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["GEN"] = GEN(model)
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
    num_classes=10,
    head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc),
    feature_layers=[
        model.conv1,
        model.block1,
        model.block2,
        model.block3,
        nn.Sequential(model.bn1, model.relu),
    ],
)

# hyperparameters determined on Textures dataset
detectors["NAC-UE"] = NACUE(
    model=model,
    layers=[model.block2, model.block3, model.bn1],
    m_bins=[200, 200, 200],
    alpha=[150.0, 200.0, 250.0],
    o_star=[25, 50, 100],
    device=device,
)

# fit detectors to training data (some require this, some do not)
print(f"> Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(
    CIFAR10(root="data", train=True, transform=trans), batch_size=128, num_workers=12
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
            for x, y in tqdm(loader, desc=dataset_name):
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
