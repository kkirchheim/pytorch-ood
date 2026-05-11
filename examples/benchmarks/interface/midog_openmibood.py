"""
OpenMIBOOD - MIDOG (Microscopy / Mitosis)
==========================================

Reproduces the MIDOG benchmark from
*OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*
(CVPR 2025).

.. note::
    The MIDOG data must be prepared first by following the
    `OpenMIBOOD setup guide <https://github.com/remic-othr/OpenMIBOOD>`_.
    The ``root`` argument should point at the directory whose subfolders match
    the relative paths in the bundled image list files (e.g. ``1a/017/...``).
"""

import pandas as pd  # additional dependency, used here for convenience
import torch
from torchvision import transforms
from torchvision.models import resnet50

from pytorch_ood.benchmark import MIDOG_OpenMIBOOD
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
loader_kwargs = {"batch_size": 64, "num_workers": 8}

# %%
# Replace this with the OpenMIBOOD pretrained MIDOG classifier (3 classes).
# See https://zenodo.org/records/14982267
model = resnet50(num_classes=3).eval().to(device)

trans = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
detectors = {
    "MSP": MaxSoftmax(model),
}

# %%
benchmark = MIDOG_OpenMIBOOD(root="data/openmibood/midog", transform=trans)

results = []
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
# This produces a table with rows for the following splits, in this order:
#
# * cs-ID:   ``midog_csid_1b``, ``midog_csid_1c`` (other scanners, same task)
# * near-OOD: ``midog_2`` ... ``midog_7`` (other scanner/staining domains)
# * far-OOD:  ``midog_ccagt`` (cervical cells), ``midog_fnac2019`` (fine-needle aspirate)
