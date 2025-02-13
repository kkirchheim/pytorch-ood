"""

ODIN - CIFAR10
==================

Reproduces the ODIN benchmark for OOD detection, from the paper
*Enhancing the reliability of out-of-distribution image detection in neural networks*.


"""

import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmark import CIFAR10_ODIN
from pytorch_ood.detector import ODIN, MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 64, "num_workers": 12}

# %%
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# %%
detectors = {
    "MSP": MaxSoftmax(model),
    "ODIN": ODIN(model, eps=0.002, norm_std=norm_std),
}

# %%
results = []
benchmark = CIFAR10_ODIN(root="data", transform=trans)

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
# +--------------------+----------+-------+---------+----------+---------+----------+
# | Dataset            | Detector | AUROC | AUTC    | AUPR-IN | AUPR-OUT | FPR95TPR |
# +====================+==========+=======+=========+=========+==========+==========+
# | TinyImageNetCrop   | MSP      | 94.59 | 33.99   | 95.77   | 93.10    | 17.18    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | TinyImageNetResize | MSP      | 88.22 | 40.12   | 89.24   | 86.00    | 42.50    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | LSUNResize         | MSP      | 91.45 | 37.91   | 92.64   | 89.46    | 29.06    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | LSUNCrop           | MSP      | 96.49 | 29.13   | 97.20   | 95.69    | 12.49    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | UniformNoise       | MSP      | 86.85 | 43.51   | 98.54   | 30.50    | 38.42    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | GaussianNoise      | MSP      | 90.29 | 41.32   | 98.99   | 36.27    | 25.69    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | TinyImageNetCrop   | ODIN     | 96.78 | 47.37   | 97.10   | 96.46    | 14.16    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | TinyImageNetResize | ODIN     | 91.44 | 48.24   | 91.45   | 91.31    | 38.84    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | LSUNResize         | ODIN     | 94.66 | 47.73   | 94.80   | 94.48    | 26.27    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | LSUNCrop           | ODIN     | 98.10 | 40.89   | 98.16   | 98.11    | 9.39     |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | UniformNoise       | ODIN     | 95.11 | 46.46   | 99.46   | 71.50    | 21.46    |
# +--------------------+----------+-------+---------+---------+----------+----------+
# | GaussianNoise      | ODIN     | 97.68 | 44.19   | 99.76   | 82.50    | 11.02    |
# +--------------------+----------+-------+---------+---------+----------+----------+
