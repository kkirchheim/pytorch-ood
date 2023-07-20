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
# +--------------------+----------+-------+---------+----------+----------+
# | Dataset            | Detector | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
# +====================+==========+=======+=========+==========+==========+
# | TinyImageNetCrop   | MSP      | 86.32 | 84.81   | 88.23    | 43.35    |
# +--------------------+----------+-------+---------+----------+----------+
# | TinyImageNetResize | MSP      | 74.64 | 70.91   | 77.29    | 65.52    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNResize         | MSP      | 75.38 | 71.16   | 78.50    | 63.36    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNCrop           | MSP      | 85.59 | 84.36   | 87.40    | 47.13    |
# +--------------------+----------+-------+---------+----------+----------+
# | Uniform            | MSP      | 77.92 | 16.86   | 97.60    | 40.44    |
# +--------------------+----------+-------+---------+----------+----------+
# | Gaussian           | MSP      | 84.78 | 23.12   | 98.41    | 30.22    |
# +--------------------+----------+-------+---------+----------+----------+
# | TinyImageNetCrop   | ODIN     | 86.89 | 84.01   | 89.02    | 40.54    |
# +--------------------+----------+-------+---------+----------+----------+
# | TinyImageNetResize | ODIN     | 80.79 | 78.44   | 82.08    | 60.10    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNResize         | ODIN     | 81.25 | 78.04   | 83.04    | 58.13    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNCrop           | ODIN     | 86.91 | 85.69   | 88.79    | 42.73    |
# +--------------------+----------+-------+---------+----------+----------+
# | Uniform            | ODIN     | 95.52 | 59.22   | 99.55    | 15.13    |
# +--------------------+----------+-------+---------+----------+----------+
# | Gaussian           | ODIN     | 98.57 | 85.14   | 99.86    | 5.76     |
# +--------------------+----------+-------+---------+----------+----------+
