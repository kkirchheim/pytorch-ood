"""

ODIN - CIFAR100
==================

Reproduces the ODIN benchmark for OOD detection, from the paper
*Enhancing the reliability of out-of-distribution image detection in neural networks*.

"""

import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmark import CIFAR100_ODIN
from pytorch_ood.detector import ODIN, MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 64}

# %%
model = WideResNet(num_classes=100, pretrained="cifar100-pt").eval().to(device)
trans = WideResNet.transform_for("cifar100-pt")
norm_std = WideResNet.norm_std_for("cifar100-pt")

# %%
detectors = {
    "MSP": MaxSoftmax(model),
    "ODIN": ODIN(model, eps=0.002, norm_std=norm_std),
}

# %%
results = []
benchmark = CIFAR100_ODIN(root="data", transform=trans)

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
# This produces a table with the following output:
#
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Dataset            | Detector | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
# +====================+==========+=======+=======+=========+==========+==========+
# | TinyImageNetCrop   | MSP      | 86.32 | 31.64 | 88.23   | 84.81    | 43.36    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetResize | MSP      | 74.64 | 40.18 | 77.29   | 70.91    | 65.56    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNResize         | MSP      | 75.38 | 39.77 | 78.50   | 71.16    | 63.36    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNCrop           | MSP      | 85.59 | 32.32 | 87.40   | 84.35    | 47.14    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Uniform            | MSP      | 77.80 | 41.27 | 97.58   | 16.76    | 40.49    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Gaussian           | MSP      | 84.97 | 35.02 | 98.43   | 23.36    | 29.45    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetCrop   | ODIN     | 86.89 | 44.10 | 89.02   | 84.01    | 40.46    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetResize | ODIN     | 80.79 | 44.87 | 82.08   | 78.43    | 60.08    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNResize         | ODIN     | 81.25 | 45.24 | 83.04   | 78.04    | 58.13    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNCrop           | ODIN     | 86.91 | 41.91 | 88.79   | 85.69    | 42.79    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Uniform            | ODIN     | 95.42 | 35.01 | 99.54   | 58.03    | 14.21    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Gaussian           | ODIN     | 98.51 | 24.27 | 99.85   | 84.87    | 5.76     |
# +--------------------+----------+-------+-------+---------+----------+----------+
