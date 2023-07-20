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
# +--------------------+----------+-------+---------+----------+----------+
# | Dataset            | Detector | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
# +====================+==========+=======+=========+==========+==========+
# | TinyImageNetCrop   | MSP      | 94.59 | 93.10   | 95.77    | 17.18    |
# +--------------------+----------+-------+---------+----------+----------+
# | TinyImageNetResize | MSP      | 88.22 | 86.01   | 89.25    | 42.56    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNResize         | MSP      | 91.45 | 89.46   | 92.65    | 29.06    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNCrop           | MSP      | 96.49 | 95.69   | 97.20    | 12.49    |
# +--------------------+----------+-------+---------+----------+----------+
# | Uniform            | MSP      | 87.22 | 29.65   | 98.60    | 32.50    |
# +--------------------+----------+-------+---------+----------+----------+
# | Gaussian           | MSP      | 90.00 | 34.68   | 98.96    | 24.75    |
# +--------------------+----------+-------+---------+----------+----------+
# | TinyImageNetCrop   | ODIN     | 96.78 | 96.46   | 97.10    | 14.17    |
# +--------------------+----------+-------+---------+----------+----------+
# | TinyImageNetResize | ODIN     | 91.44 | 91.31   | 91.45    | 38.83    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNResize         | ODIN     | 94.66 | 94.48   | 94.80    | 26.25    |
# +--------------------+----------+-------+---------+----------+----------+
# | LSUNCrop           | ODIN     | 98.10 | 98.11   | 98.16    | 9.37     |
# +--------------------+----------+-------+---------+----------+----------+
# | Uniform            | ODIN     | 95.94 | 73.27   | 99.57    | 17.16    |
# +--------------------+----------+-------+---------+----------+----------+
# | Gaussian           | ODIN     | 97.02 | 80.50   | 99.67    | 12.89    |
# +--------------------+----------+-------+---------+----------+----------+
