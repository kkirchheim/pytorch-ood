"""

CIFAR 10 ODIN Benchmark
==============================

+--------------------+----------+-------+---------+----------+----------+
| Dataset            | Detector | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+====================+==========+=======+=========+==========+==========+
| TinyImageNetCrop   | MSP      | 94.59 | 93.10   | 95.77    | 17.20    |
+--------------------+----------+-------+---------+----------+----------+
| TinyImageNetResize | MSP      | 88.22 | 86.01   | 89.25    | 42.53    |
+--------------------+----------+-------+---------+----------+----------+
| LSUNResize         | MSP      | 91.45 | 89.46   | 92.65    | 29.05    |
+--------------------+----------+-------+---------+----------+----------+
| LSUNCrop           | MSP      | 96.49 | 95.69   | 97.20    | 12.49    |
+--------------------+----------+-------+---------+----------+----------+
| TinyImageNetCrop   | ODIN     | 96.78 | 96.46   | 97.10    | 14.14    |
+--------------------+----------+-------+---------+----------+----------+
| TinyImageNetResize | ODIN     | 91.44 | 91.31   | 91.45    | 38.79    |
+--------------------+----------+-------+---------+----------+----------+
| LSUNResize         | ODIN     | 94.66 | 94.48   | 94.80    | 26.24    |
+--------------------+----------+-------+---------+----------+----------+
| LSUNCrop           | ODIN     | 98.10 | 98.12   | 98.17    | 9.34     |
+--------------------+----------+-------+---------+----------+----------+


"""
import pandas as pd  # additional dependency, used here for convenience
import torch

from pytorch_ood.benchmarks import CIFAR10_ODIN
from pytorch_ood.detector import ODIN, MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 64}

# %%
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)
trans = WideResNet.transform_for("cifar10-pt")

# %%
detectors = {
    "MSP": MaxSoftmax(model),
    "ODIN": ODIN(model, eps=0.002, norm_std=[x / 255 for x in [63.0, 62.1, 66.7]]),
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
