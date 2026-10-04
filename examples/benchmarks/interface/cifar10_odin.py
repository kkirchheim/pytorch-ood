"""

ODIN - CIFAR10
==================

Reproduces the ODIN benchmark for OOD detection, from the paper
*Enhancing the reliability of out-of-distribution image detection in neural networks*.


"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/benchmarks.png"

import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmark import CIFAR10_ODIN
from pytorch_ood.detector import ODIN, MaxSoftmax
from pytorch_ood.model import get_model_info, load_model, load_transform
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 64, "num_workers": 12}

# %%
model = load_model("wrn-40-2/cifar10/crossentropy").to(device)
trans = load_transform("wrn-40-2/cifar10/crossentropy")
norm_std = get_model_info("wrn-40-2/cifar10/crossentropy").preprocessing.std

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
# +--------------------+----------+-------+-------+---------+----------+----------+
# | Dataset            | Detector | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
# +====================+==========+=======+=======+=========+==========+==========+
# | TinyImageNetCrop   | MSP      | 94.59 | 33.98 | 95.77   | 93.10    | 17.18    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetResize | MSP      | 88.04 | 40.08 | 89.11   | 85.76    | 43.40    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNResize         | MSP      | 91.08 | 38.05 | 92.27   | 89.06    | 30.39    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNCrop           | MSP      | 96.49 | 29.13 | 97.20   | 95.69    | 12.49    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | UniformNoise       | MSP      | 86.84 | 43.52 | 98.54   | 30.49    | 38.45    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | GaussianNoise      | MSP      | 90.29 | 41.33 | 98.99   | 36.26    | 25.68    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetCrop   | ODIN     | 97.98 | 35.38 | 98.08   | 97.95    | 10.03    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | TinyImageNetResize | ODIN     | 90.34 | 39.68 | 90.19   | 90.49    | 43.38    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNResize         | ODIN     | 94.88 | 37.59 | 94.84   | 94.99    | 26.50    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | LSUNCrop           | ODIN     | 98.86 | 32.04 | 98.84   | 98.93    | 5.49     |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | UniformNoise       | ODIN     | 91.78 | 38.79 | 99.06   | 52.68    | 36.70    |
# +--------------------+----------+-------+-------+---------+----------+----------+
# | GaussianNoise      | ODIN     | 96.24 | 36.68 | 99.61   | 66.31    | 12.50    |
# +--------------------+----------+-------+-------+---------+----------+----------+
