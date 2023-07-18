"""
OpenOOD - ImageNet
===================

Reproduces the OpenOOD benchmark for OOD detection, using a pre-trained ResNet 50.

.. warning :: This is currently incomplete, see :class:`ImageNet-OpenOOD <pytorch_ood.benchmark.ImageNet_OpenOOD>`.

"""
import pandas as pd  # additional dependency, used here for convenience
import torch
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from pytorch_ood.benchmark import ImageNet_OpenOOD
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0"
loader_kwargs = {"batch_size": 16, "num_workers": 12}

# %%
model = resnet50(ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
trans = ResNet50_Weights.IMAGENET1K_V1.transforms()

print(trans)

# %%
# If you want to test more detectors, you can just add them here
detectors = {
    "MSP": MaxSoftmax(model),
}

# %%
# The ImageNet root should contain at least the validation tar, the dev kit tar, and the meta.bin
# that is generated by the torchvision ImageNet implementation.
results = []
benchmark = ImageNet_OpenOOD(root="data", image_net_root="data/imagenet-2012/", transform=trans)


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
# This should produce a table with the following output:
#
# +-------------+----------+-------+---------+----------+----------+
# | Dataset     | Detector | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
# +=============+==========+=======+=========+==========+==========+
# | ImageNetO   | MSP      | 28.64 | 2.52    | 94.85    | 91.20    |
# +-------------+----------+-------+---------+----------+----------+
# | OpenImagesO | MSP      | 84.98 | 62.61   | 94.67    | 49.95    |
# +-------------+----------+-------+---------+----------+----------+
# | Textures    | MSP      | 80.46 | 37.50   | 96.80    | 67.75    |
# +-------------+----------+-------+---------+----------+----------+
# | SVHN        | MSP      | 97.62 | 95.56   | 98.77    | 11.58    |
# +-------------+----------+-------+---------+----------+----------+
# | MNIST       | MSP      | 90.04 | 90.45   | 89.88    | 39.03    |
# +-------------+----------+-------+---------+----------+----------+
#