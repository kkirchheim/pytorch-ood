"""

CIFAR 10
==============================

Example benchmark code for CIFAR10

+------------------+-------+-------+---------+----------+----------+
| Detector         | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
+==================+=======+=======+=========+==========+==========+
| MultiMahalanobis | 93.70 | 44.00 | 87.78   | 96.55    | 21.32    |
+------------------+-------+-------+---------+----------+----------+
| GEN              | 93.32 | 29.78 | 87.75   | 94.56    | 29.80    |
+------------------+-------+-------+---------+----------+----------+
| EnergyBased      | 93.07 | 35.47 | 87.03   | 94.41    | 31.43    |
+------------------+-------+-------+---------+----------+----------+
| ASH              | 93.01 | 35.73 | 86.95   | 94.37    | 31.58    |
+------------------+-------+-------+---------+----------+----------+
| KNN              | 93.00 | 32.01 | 87.99   | 94.27    | 28.00    |
+------------------+-------+-------+---------+----------+----------+
| MaxLogit         | 93.00 | 35.88 | 86.96   | 94.35    | 31.58    |
+------------------+-------+-------+---------+----------+----------+
| Gram             | 92.75 | 47.17 | 86.62   | 97.15    | 40.54    |
+------------------+-------+-------+---------+----------+----------+
| Mahalanobis+ODIN | 92.55 | 42.54 | 86.78   | 94.99    | 27.19    |
+------------------+-------+-------+---------+----------+----------+
| RMD              | 92.53 | 31.33 | 87.11   | 93.72    | 28.37    |
+------------------+-------+-------+---------+----------+----------+
| DICE             | 92.24 | 37.48 | 85.42   | 94.24    | 33.76    |
+------------------+-------+-------+---------+----------+----------+
| ViM              | 92.23 | 40.32 | 85.69   | 94.83    | 29.56    |
+------------------+-------+-------+---------+----------+----------+
| Entropy          | 91.97 | 35.95 | 86.64   | 93.40    | 29.97    |
+------------------+-------+-------+---------+----------+----------+
| Mahalanobis      | 91.74 | 42.79 | 86.15   | 93.70    | 28.80    |
+------------------+-------+-------+---------+----------+----------+
| fDBD             | 91.69 | 35.32 | 83.40   | 93.86    | 36.39    |
+------------------+-------+-------+---------+----------+----------+
| MSP              | 91.35 | 37.07 | 86.31   | 92.35    | 30.21    |
+------------------+-------+-------+---------+----------+----------+
| ODIN             | 91.21 | 37.98 | 83.58   | 93.92    | 36.38    |
+------------------+-------+-------+---------+----------+----------+
| GMM              | 90.79 | 42.46 | 86.43   | 91.01    | 28.93    |
+------------------+-------+-------+---------+----------+----------+
| SHE              | 90.08 | 39.69 | 82.38   | 92.90    | 38.38    |
+------------------+-------+-------+---------+----------+----------+
| KLMatching       | 88.38 | 39.24 | 72.08   | 91.27    | 57.69    |
+------------------+-------+-------+---------+----------+----------+
| GradUncertainty  | 88.30 | 39.23 | 78.90   | 91.90    | 44.96    |
+------------------+-------+-------+---------+----------+----------+
| NAC-UE           | 88.00 | 40.75 | 80.74   | 89.46    | 47.65    |
+------------------+-------+-------+---------+----------+----------+
| GradNorm         | 78.46 | 41.97 | 68.64   | 85.85    | 56.19    |
+------------------+-------+-------+---------+----------+----------+
| RankFeat         | 44.58 | 50.07 | 37.80   | 59.16    | 92.20    |
+------------------+-------+-------+---------+----------+----------+



"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/benchmarks.png"

import pandas as pd  # additional dependency, used here for convenience
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from tqdm.auto import tqdm  # additional dependency, used here for convenience

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Places365,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
)
from pytorch_ood.detector import (
    ASH,
    DICE,
    GEN,
    GMM,
    KNN,
    NACUE,
    ODIN,
    RMD,
    SHE,
    EnergyBased,
    Entropy,
    GradNorm,
    GradUncertainty,
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

# %%
# Setup preprocessing
trans = load_transform("wrn-40-2/cifar10/crossentropy")
norm_std = get_model_info("wrn-40-2/cifar10/crossentropy").preprocessing.std

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
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)

# %%
# **Stage 2**: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}

detectors["KNN"] = KNN(model.features)
detectors["GMM"] = GMM(model.features)
detectors["fDBD"] = fDBD(encoder=model.features, head=model.fc)

detectors["ASH"] = ASH(backbone=model.feature_maps, head=model.forward_feature_maps)
detectors["RankFeat"] = RankFeat(backbone=model.feature_maps, head=model.forward_feature_maps)

detectors["GradUncertainty"] = GradUncertainty(
    model, param_filter=lambda name: name.startswith("fc")
)
detectors["GradNorm"] = GradNorm(model, param_filter=lambda name: name == "fc.weight")

detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis+ODIN"] = MahalanobisODIN(model.features, norm_std=norm_std, eps=0.002)
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
