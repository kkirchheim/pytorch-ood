"""

CIFAR 100
==============================

The evaluation is the same as for CIFAR 10.

+------------------+-------+---------+----------+----------+
| Detector         | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+==================+=======+=========+==========+==========+
| SHE              | 59.43 | 77.44   | 68.37    | 100.00   |
+------------------+-------+---------+----------+----------+
| Mahalanobis      | 75.35 | 81.59   | 65.62    | 58.87    |
+------------------+-------+---------+----------+----------+
| MSP              | 78.78 | 82.37   | 71.34    | 57.67    |
+------------------+-------+---------+----------+----------+
| Mahalanobis+ODIN | 79.24 | 84.58   | 68.69    | 55.91    |
+------------------+-------+---------+----------+----------+
| KLMatching       | 79.88 | 83.53   | 68.23    | 60.02    |
+------------------+-------+---------+----------+----------+
| ODIN             | 80.80 | 83.96   | 73.40    | 54.92    |
+------------------+-------+---------+----------+----------+
| Entropy          | 81.19 | 84.61   | 73.08    | 56.49    |
+------------------+-------+---------+----------+----------+
| ViM              | 81.73 | 85.87   | 72.91    | 49.86    |
+------------------+-------+---------+----------+----------+
| RMD              | 83.23 | 86.94   | 74.56    | 50.55    |
+------------------+-------+---------+----------+----------+
| MaxLogit         | 84.70 | 86.66   | 78.33    | 47.40    |
+------------------+-------+---------+----------+----------+
| EnergyBased      | 85.00 | 86.88   | 78.69    | 46.70    |
+------------------+-------+---------+----------+----------+
| DICE             | 85.35 | 87.32   | 78.99    | 46.17    |
+------------------+-------+---------+----------+----------+

"""
import pandas as pd  # additional dependency, used here for convenience
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    Places365,
    TinyImageNet,
)
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM, RMD, DICE, SHE,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

device = "cuda:0"

fix_random_seed(123)

# setup preprocessing
trans = WideResNet.transform_for("cifar100-pt")
norm_std = WideResNet.norm_std_for("cifar100-pt")

# %%
# Setup datasets
dataset_in_test = CIFAR100(root="data", train=False, transform=trans, download=True)

# create all OOD datasets
ood_datasets = [Textures, TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize, Places365, CIFAR10, MNIST, FashionMNIST]
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
model = WideResNet(num_classes=100, pretrained="cifar100-pt").eval().to(device)

# Stage 2: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}
detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis+ODIN"] = Mahalanobis(model.features, norm_std=norm_std, eps=0.002)
detectors["Mahalanobis"] = Mahalanobis(model.features)
detectors["KLMatching"] = KLMatching(model)
detectors["SHE"] = SHE(model.features, model.fc)
detectors["MSP"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)
detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)
detectors["DICE"] = DICE(model=model.features, w=model.fc.weight, b=model.fc.bias, p=0.65)
detectors["RMD"] = RMD(model.features)

# %%
# **Stage 2**: fit detectors to training data (some require this, some do not)
print(f"> Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(CIFAR100(root="data", train=True, transform=trans), batch_size=256, num_workers=12)
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
