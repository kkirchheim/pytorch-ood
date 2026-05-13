"""

OpenOOD v1.5 - CIFAR10
========================

Reproduces the OpenOOD v1.5 benchmark for OOD detection on CIFAR-10, using the WideResNet
model from the Hendrycks baseline paper.

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
# This should produce a table with results for the following OOD datasets:
#
# Near-OOD:
# * CIFAR100
# * TinyImageNet
#
# Far-OOD:
# * MNIST
# * SVHN
# * Textures
# * Places365
#
