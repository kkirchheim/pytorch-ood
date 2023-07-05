"""

CIFAR 100
==============================

The evaluation is the same as for CIFAR 10.



+-------------+-------+---------+----------+----------+
| Detector    | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=========+==========+==========+
| MaxSoftmax  | 79.10 | 73.75   | 82.92    | 58.16    |
+-------------+-------+---------+----------+----------+
| KLMatching  | 80.47 | 75.53   | 82.70    | 57.91    |
+-------------+-------+---------+----------+----------+
| ODIN        | 81.46 | 76.65   | 84.88    | 55.57    |
+-------------+-------+---------+----------+----------+
| Entropy     | 81.52 | 77.09   | 84.38    | 57.12    |
+-------------+-------+---------+----------+----------+
| Mahalanobis | 83.91 | 79.86   | 86.08    | 46.99    |
+-------------+-------+---------+----------+----------+
| MaxLogit    | 84.64 | 79.95   | 87.25    | 48.51    |
+-------------+-------+---------+----------+----------+
| EnergyBased | 84.90 | 80.27   | 87.46    | 47.85    |
+-------------+-------+---------+----------+----------+
| ViM         | 85.87 | 81.18   | 88.81    | 41.83    |
+-------------+-------+---------+----------+----------+

"""
import pandas as pd  # additional dependency, used here for convenience
import torch
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
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
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
detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis"] = Mahalanobis(model.features, norm_std=norm_std, eps=0.002)
detectors["KLMatching"] = KLMatching(model)
detectors["MaxSoftmax"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)
detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)

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
