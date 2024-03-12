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
from pytorch_ood.detector import ASH, MaxSoftmax, ReAct
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
# +--------------+----------+-------+---------+----------+----------+
# | Dataset      | Detector | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
# +==============+==========+=======+=========+==========+==========+
# | CIFAR100     | MSP      | 87.83 | 85.20   | 88.42    | 43.08    |
# +--------------+----------+-------+---------+----------+----------+
# | TinyImageNet | MSP      | 87.06 | 85.05   | 86.82    | 51.27    |
# +--------------+----------+-------+---------+----------+----------+
# | MNIST        | MSP      | 92.66 | 90.29   | 94.33    | 22.47    |
# +--------------+----------+-------+---------+----------+----------+
# | FashionMNIST | MSP      | 94.95 | 93.36   | 96.18    | 15.59    |
# +--------------+----------+-------+---------+----------+----------+
# | Textures     | MSP      | 88.51 | 78.50   | 92.99    | 40.86    |
# +--------------+----------+-------+---------+----------+----------+
# | Places365    | MSP      | 88.24 | 95.61   | 71.17    | 44.65    |
# +--------------+----------+-------+---------+----------+----------+
#
