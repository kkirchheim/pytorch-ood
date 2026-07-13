"""

OpenOOD v1.5 - CIFAR10
========================

Reproduces the OpenOOD v1.5 benchmark for OOD detection on CIFAR-10, using the WideResNet
model from the Hendrycks baseline paper.

"""

import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmark import CIFAR10_OpenOOD
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.model import load_model, load_transform

device = "cuda:0"
loader_kwargs = {"batch_size": 64}

# %%
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)
trans = load_transform("wrn-40-2/cifar10/crossentropy")

# %%
# Just add more detectors here if you want to test more
detectors = {
    "MSP": MaxSoftmax(model),
}

# %%
results = []
benchmark = CIFAR10_OpenOOD(root="data", transform=trans)

with torch.no_grad():
    for detector_name, detector in detectors.items():
        print(f"> Evaluating {detector_name}")
        res = benchmark.evaluate(detector, loader_kwargs=loader_kwargs, device=device)
        for r in res:
            r.update({"Detector": detector_name})
        results += res

df = pd.DataFrame(results)
print((df.set_index(["Dataset", "Detector"]) * 100).to_csv(float_format="%.2f"))

# %%
# This should produces the following table:
#
# +---------------+----------+-------+-------+---------+----------+----------+
# | Dataset       | Detector | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
# +===============+==========+=======+=======+=========+==========+==========+
# | CIFAR100      | MSP      | 87.83 | 40.69 | 88.42   | 85.20    | 43.04    |
# +---------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNet  | MSP      | 87.01 | 40.57 | 86.54   | 85.05    | 51.33    |
# +---------------+----------+-------+-------+---------+----------+----------+
# | MNIST         | MSP      | 92.66 | 37.24 | 94.32   | 90.29    | 22.47    |
# +---------------+----------+-------+-------+---------+----------+----------+
# | SVHN          | MSP      | 91.91 | 36.89 | 86.50   | 95.81    | 28.47    |
# +---------------+----------+-------+-------+---------+----------+----------+
# | Textures      | MSP      | 88.51 | 39.69 | 93.00   | 78.50    | 41.30    |
# +---------------+----------+-------+-------+---------+----------+----------+
# | Places365     | MSP      | 88.25 | 39.97 | 71.19   | 95.61    | 44.55    |
# +---------------+----------+-------+-------+---------+----------+----------+
#
