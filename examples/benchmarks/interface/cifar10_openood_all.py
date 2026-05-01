"""
OpenOOD v1.5 - CIFAR10, Many Detectors
=======================================

Evaluates a broad set of image-classification detectors on the CIFAR-10 OpenOOD
benchmark using the benchmark interface and cached intermediate representations.

This example focuses on detectors that can run directly on the pretrained
WideResNet used throughout the repository. It omits methods that require extra
external dependencies or method-specific trained weights, such as OpenMax and
WeightedEBO.
"""

from collections import OrderedDict
from copy import deepcopy

import pandas as pd  # additional dependency, used here for convenience
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from pytorch_ood.benchmark import CIFAR10_OpenOOD
from pytorch_ood.detector import (
    ASH,
    DICE,
    EnergyBased,
    Entropy,
    GEN,
    GMM,
    GradNorm,
    GradNormKL,
    Gram,
    KLMatching,
    KNN,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    MCD,
    MultiMahalanobis,
    NACUE,
    NCI,
    NNGuide,
    ODIN,
    PNML,
    RMD,
    RankFeat,
    ReAct,
    SHE,
    TemperatureScaling,
    ViM,
    VRA,
    fDBD,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import fix_random_seed

fix_random_seed(123)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
loader_kwargs = {"batch_size": 128, "num_workers": 12}
cache_dir = "data/benchmark-cache"
cache_key = "cifar10-openood-wrn-cifar10-pt"
react_threshold = 1.0


def build_detectors(model, norm_std, react_threshold):
    detectors = OrderedDict()

    detectors["MSP"] = MaxSoftmax(model)
    detectors["TemperatureScaling"] = TemperatureScaling(model)
    detectors["Entropy"] = Entropy(model)
    detectors["EnergyBased"] = EnergyBased(model)
    detectors["MaxLogit"] = MaxLogit(model)
    detectors["GEN"] = GEN(model)
    detectors["KLMatching"] = KLMatching(model)
    detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)
    detectors["MCD"] = MCD(model, samples=30, mode="var")

    detectors["KNN"] = KNN(model.features)
    detectors["GMM"] = GMM(model.features)
    detectors["PNML"] = PNML(model.features, model.fc)
    detectors["NNGuide"] = NNGuide(model.features, model.fc)
    detectors["fDBD"] = fDBD(encoder=model.features, head=model.fc)
    detectors["Mahalanobis"] = Mahalanobis(model.features)
    detectors["Mahalanobis+ODIN"] = Mahalanobis(model.features, norm_std=norm_std, eps=0.002)
    detectors["RMD"] = RMD(model.features)
    detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
    detectors["NCI"] = NCI(encoder=model.features, head=model.fc, alpha=0.0)
    detectors["SHE"] = SHE(model.features, model.fc)
    detectors["DICE"] = DICE(encoder=model.features, w=model.fc.weight, b=model.fc.bias, p=65.0)
    detectors["ReAct"] = ReAct(model.features, model.fc, threshold=react_threshold)
    detectors["VRA"] = VRA(model.features, model.fc)

    detectors["ASH"] = ASH(
        backbone=model.feature_maps,
        head=model.forward_feature_maps,
    )
    detectors["RankFeat"] = RankFeat(
        backbone=model.feature_maps,
        head=model.forward_feature_maps,
    )

    detectors["MultiMahalanobis"] = MultiMahalanobis(
        [
            model.conv1,
            model.block1,
            model.block2,
            model.block3,
            nn.Sequential(model.bn1, model.relu),
        ]
    )
    detectors["Gram"] = Gram(
        num_classes=10,
        head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc),
        feature_layers=[
            model.conv1,
            model.block1,
            model.block2,
            model.block3,
            nn.Sequential(model.bn1, model.relu),
        ],
    )

    model_gn = deepcopy(model)
    model_gn.requires_grad_(False)
    model_gn.fc.requires_grad_(True)
    detectors["GradNorm"] = GradNorm(model_gn, param_filter=lambda name: name.startswith("fc"))

    model_gnkl = deepcopy(model)
    model_gnkl.requires_grad_(False)
    model_gnkl.fc.requires_grad_(True)
    detectors["GradNormKL"] = GradNormKL(
        model_gnkl, param_filter=lambda name: name.startswith("fc")
    )

    detectors["NAC-UE"] = NACUE(
        model=model,
        layers=[model.block2, model.block3, model.bn1],
        m_bins=[200, 200, 200],
        alpha=[150.0, 200.0, 250.0],
        o_star=[25, 50, 100],
        device=device,
    )

    return detectors


def fit_detectors(detectors, train_loader, calibration_loader):
    for detector_name, detector in detectors.items():
        if not getattr(detector, "requires_fit", False):
            continue

        fit_loader = (
            calibration_loader
            if detector_name in {"TemperatureScaling", "KLMatching"}
            else train_loader
        )
        print(f"--> Fitting {detector_name}")
        detector.to(device)
        detector.fit(fit_loader)


# %%
print("STAGE 1: Creating model and benchmark")
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")
benchmark = CIFAR10_OpenOOD(root="data", transform=trans)

train_dataset = benchmark.train_set()
train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
calibration_loader = DataLoader(
    Subset(train_dataset, range(len(train_dataset) - 5000, len(train_dataset))),
    shuffle=False,
    **loader_kwargs,
)

print("STAGE 2: Creating and fitting detectors")
detectors = build_detectors(model=model, norm_std=norm_std, react_threshold=react_threshold)
fit_detectors(
    detectors=detectors, train_loader=train_loader, calibration_loader=calibration_loader
)

# %%
print("STAGE 3: Evaluating detectors")
results = []

for detector_name, detector in detectors.items():
    print(f"> Evaluating {detector_name}")
    res = benchmark.evaluate(
        detector,
        loader_kwargs=loader_kwargs,
        device=device,
        cache=True,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    for row in res:
        row.update({"Detector": detector_name})
    results += res

df = pd.DataFrame(results)
print((df.set_index(["Dataset", "Detector"]) * 100).to_csv(float_format="%.2f"))

print("\nMean scores:")
mean_scores = df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean()
print((mean_scores.sort_values("AUROC", ascending=False) * 100).to_csv(float_format="%.2f"))
