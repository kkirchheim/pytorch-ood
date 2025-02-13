"""

OpenOOD - CIFAR10
==================

Reproduces the OpenOOD benchmark for OOD detection, using the WideResNet
model from the Hendrycks baseline paper.

.. warning :: This is currently incomplete, see :class:`CIFAR10-OpenOOD <pytorch_ood.benchmark.CIFAR10_OpenOOD>`.

"""

import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmark import CIFAR10_OpenOOD
from pytorch_ood.detector import MaxSoftmax, ReAct, ASH
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 64}

# %%
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

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
# This should produce the following table:
#
# +--------------+----------+-------+-------+---------+----------+----------+
# | Dataset      | Detector | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
# +==============+==========+=======+=======+=========+==========+==========+
# | CIFAR100     | MSP      | 87.82 | 40.69 | 88.42   | 85.20    | 43.09    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNet | MSP      | 86.99 | 40.65 | 86.48   | 85.07    | 51.52    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | MNIST        | MSP      | 92.66 | 37.23 | 94.33   | 90.30    | 22.46    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | FashionMNIST | MSP      | 94.95 | 33.53 | 96.18   | 93.36    | 15.58    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | Textures     | MSP      | 88.51 | 39.68 | 92.99   | 78.50    | 40.89    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | Places365    | MSP      | 88.24 | 39.93 | 71.17   | 95.61    | 44.63    |
# +--------------+----------+-------+-------+---------+----------+----------+
