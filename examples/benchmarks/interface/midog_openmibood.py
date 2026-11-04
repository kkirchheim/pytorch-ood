"""
OpenMIBOOD - MIDOG (Microscopy / Mitosis)
==========================================

Reproduces the MIDOG benchmark from
*OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*
(CVPR 2025) with the pre-trained classifier and the detector hyperparameters of the paper.

.. note::
    With ``download=True``, the MIDOG++ slides (65 GB) and CCAgT are downloaded and cut into
    patches, which takes several hours. FNAC 2019 can not be downloaded automatically, see
    :class:`~pytorch_ood.benchmark.MIDOG_OpenMIBOOD`. With ``skip_missing=True``, the benchmark is
    evaluated without it.
"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/benchmarks.png"

import pandas as pd  # additional dependency, used here for convenience
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from pytorch_ood.benchmark import MIDOG_OpenMIBOOD
from pytorch_ood.detector import (
    ASH,
    DICE,
    GEN,
    KNN,
    EnergyBased,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    NNGuide,
    RankFeat,
    ReAct,
    ViM,
    fDBD,
)
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
loader_kwargs = {"batch_size": 64, "num_workers": 4}

# %%
# Load the ResNet-50 trained by the OpenMIBOOD authors on the MIDOG patches (3 classes).
# See https://zenodo.org/records/14982267
model = resnet50(num_classes=3)
state_dict = torch.hub.load_state_dict_from_url(
    "https://zenodo.org/records/14982267/files/midog_classifier.pth?download=1",
    map_location="cpu",
    file_name="midog_classifier.pth",
)
model.load_state_dict(state_dict)
model = model.eval().to(device)

# pooled 2048-dimensional features, used by the feature-based detectors
encoder = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
# ReAct and ASH shape the pooled features, passed as feature maps of shape B x 2048 x 1 x 1
pooled = torch.nn.Sequential(*list(model.children())[:-1])
pooled_head = torch.nn.Sequential(torch.nn.Flatten(), model.fc)
# RankFeat operates on the feature maps of the last block
backbone = torch.nn.Sequential(*list(model.children())[:-2])
head = torch.nn.Sequential(model.avgpool, torch.nn.Flatten(), model.fc)

# The patches are 50x50 pixels, normalized with the MIDOG statistics of the paper
trans = transforms.Compose(
    [
        transforms.Resize(50),
        transforms.CenterCrop(50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.712, 0.496, 0.756], std=[0.167, 0.167, 0.110]),
    ]
)

# %%
benchmark = MIDOG_OpenMIBOOD(
    root="data/openmibood/midog", transform=trans, download=True, skip_missing=True
)
train_loader = DataLoader(benchmark.train_set(), **loader_kwargs)
# OpenMIBOOD fits KLMatching and ReAct on the ID validation set
val_loader = DataLoader(benchmark.val_set(), **loader_kwargs)

# %%
# Detectors with the hyperparameters that OpenMIBOOD tuned on the MIDOG validation set.
#
# With 1,896 training samples and 2,048 features, the covariance matrix is singular, so
# Mahalanobis uses its pseudo-inverse.
#
# OpenMIBOOD builds the NNGuide bank from a random half of the training set. Here, the bank
# holds all training samples, so the results differ by up to one point from the paper.
#
# OpenMIBOOD applies RankFeat to the feature maps of both the third and the fourth block and
# averages the logits. :class:`~pytorch_ood.detector.RankFeat` only uses the last block, so its
# results differ from the paper.
detectors = {
    "MSP": MaxSoftmax(model),
    "GEN": GEN(model, gamma=0.01, M=2),
    "KLMatching": KLMatching(model).fit(val_loader),
    "MaxLogit": MaxLogit(model),
    "Energy": EnergyBased(model, t=2.0),
    "Mahalanobis": Mahalanobis(encoder, use_pinv=True).fit(train_loader),
    "ViM": ViM(encoder, d=128, w=model.fc.weight, b=model.fc.bias).fit(train_loader),
    "KNN": KNN(encoder, k=5, normalize=True).fit(train_loader),
    "NNGuide": NNGuide(encoder, model.fc, k=5).fit(train_loader),
    "fDBD": fDBD(encoder, model.fc, distance_as_normalizer=False).fit(train_loader),
    "DICE": DICE(encoder, w=model.fc.weight, b=model.fc.bias, p=0.85).fit(train_loader),
    "ASH": ASH(pooled, pooled_head, percentile=0.65),
    "ReAct": ReAct(pooled, pooled_head, percentile=0.95).fit(val_loader),
    "RankFeat": RankFeat(backbone, head, detector=lambda z: EnergyBased.score(z, t=0.1)),
}

# %%
results = []
with torch.no_grad():
    for name, detector in detectors.items():
        print(f"Evaluating {name}")
        res = benchmark.evaluate(detector, loader_kwargs=loader_kwargs, device=device)
        for r in res:
            r.update({"Detector": name})
        results += res

# %%
# AUROC in percent, with one row per test set and one column per detector:
df = pd.DataFrame(results)
auroc = df.pivot(index="Dataset", columns="Detector", values="AUROC") * 100
print(auroc.to_string(float_format="%.2f"))

# %%
# This produces a table with rows for the following test sets:
#
# * covariate-shifted ID: ``midog_csid_1b``, ``midog_csid_1c`` (other scanners, same task)
# * near-OOD: ``midog_2`` ... ``midog_7`` (other cancer types and species)
# * far-OOD: ``midog_ccagt`` (cervical cells), ``midog_fnac2019`` (fine-needle aspiration
#   cytology, only if prepared manually)
