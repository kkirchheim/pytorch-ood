"""

ODIN - CIFAR100
==================

Reproduces the ODIN benchmark for OOD detection, from the paper
*Enhancing the reliability of out-of-distribution image detection in neural networks*.

"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/benchmarks.png"

import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmark import CIFAR100_ODIN
from pytorch_ood.detector import ODIN, MaxSoftmax
from pytorch_ood.model import get_model_info, load_model, load_transform
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 64}

# %%
model = load_model("wrn-40-2/cifar100/crossentropy").to(device)
trans = load_transform("wrn-40-2/cifar100/crossentropy")
norm_std = get_model_info("wrn-40-2/cifar100/crossentropy").preprocessing.std

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
# | TinyImageNetCrop   | MSP      | 86.32 | 31.64 | 88.23   | 84.81    | 43.33    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetResize | MSP      | 73.98 | 40.72 | 76.78   | 70.15    | 66.06    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNResize         | MSP      | 74.12 | 40.76 | 77.33   | 69.91    | 64.96    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNCrop           | MSP      | 85.59 | 32.32 | 87.40   | 84.36    | 47.13    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Uniform            | MSP      | 77.85 | 41.34 | 97.60   | 16.73    | 39.66    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Gaussian           | MSP      | 85.25 | 34.72 | 98.46   | 23.89    | 30.45    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetCrop   | ODIN     | 95.56 | 38.90 | 96.03   | 95.09    | 19.67    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetResize | ODIN     | 84.10 | 42.39 | 84.47   | 82.90    | 56.35    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNResize         | ODIN     | 84.52 | 42.19 | 85.40   | 82.94    | 55.23    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNCrop           | ODIN     | 96.38 | 38.22 | 96.80   | 96.09    | 16.52    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Uniform            | ODIN     | 77.47 | 44.66 | 97.54   | 16.51    | 42.01    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Gaussian           | ODIN     | 93.32 | 39.99 | 99.32   | 45.56    | 18.16    |
# +--------------------+----------+-------+-------+---------+----------+----------+
