"""

OpenOOD v1.5 - CIFAR10
========================

Reproduces the OpenOOD v1.5 benchmark for OOD detection on CIFAR-10, using the WideResNet
model from the Hendrycks baseline paper.

"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/benchmarks.png"

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
# +--------------+----------+-------+-------+---------+----------+----------+
# | Dataset      | Detector | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
# +==============+==========+=======+=======+=========+==========+==========+
# | CIFAR100     | MSP      | 87.91 | 40.67 | 88.46   | 85.25    | 43.02    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNet | MSP      | 89.29 | 39.77 | 91.06   | 85.22    | 36.78    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | MNIST        | MSP      | 93.31 | 36.96 | 83.42   | 98.71    | 20.04    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | SVHN         | MSP      | 91.91 | 37.67 | 86.28   | 95.95    | 25.04    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | Textures     | MSP      | 88.57 | 39.69 | 92.40   | 80.13    | 41.16    |
# +--------------+----------+-------+-------+---------+----------+----------+
# | Places365    | MSP      | 89.23 | 39.68 | 74.29   | 96.15    | 37.89    |
# +--------------+----------+-------+-------+---------+----------+----------+
#
