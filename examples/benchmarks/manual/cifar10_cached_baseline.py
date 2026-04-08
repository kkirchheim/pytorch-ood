"""

CIFAR 10 Cached Baseline
========================

Benchmark code for CIFAR-10 that uses the benchmark-level caching support
to reuse logits and pooled features across several detector evaluations.
Cached representations are kept on the benchmark object and can optionally
be persisted to disk between Python sessions via ``cache_dir`` and
``cache_key``.

"""

from copy import deepcopy
import pandas as pd  # additional dependency, used here for convenience
from torch import nn

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
    MultiMahalanobis,
    NACUE,
    ODIN,
    RMD,
    RankFeat,
    SHE,
    ViM,
    fDBD,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import fix_random_seed

device = "cuda:0"
loader_kwargs = {"batch_size": 128, "num_workers": 12}
cache_dir = "data/benchmark-cache"
cache_key = "cifar10-openood-wrn-cifar10-pt"

fix_random_seed(123)


# %%
# Setup preprocessing
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# %%
# Stage 1: Create model
print("STAGE 1: Creating a Model")
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)

# %%
# Stage 2: Create detectors
print("STAGE 2: Creating OOD Detectors")
detectors = {}

detectors["KNN"] = KNN(model.features)
detectors["GMM"] = GMM(model.features)
detectors["fDBD"] = fDBD(encoder=model.features, head=model.fc)

detectors["ASH"] = ASH(backbone=model.features_before_pool, head=model.forward_from_before_pool)
detectors["RankFeat"] = RankFeat(
    backbone=model.features_before_pool, head=model.forward_from_before_pool
)

model_gn = deepcopy(model)
model_gn.requires_grad_(False)
model_gn.fc.requires_grad_(True)
detectors["GradNorm"] = GradNorm(model_gn, param_filter=lambda name: name.startswith("fc"))

model_gnkl = deepcopy(model)
model_gnkl.requires_grad_(False)
model_gnkl.fc.requires_grad_(True)
detectors["GradNormKL"] = GradNormKL(model_gnkl, param_filter=lambda name: name.startswith("fc"))

detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis+ODIN"] = Mahalanobis(model.features, norm_std=norm_std, eps=0.002)
detectors["Mahalanobis"] = Mahalanobis(model.features)

detectors["KLMatching"] = KLMatching(model)
detectors["SHE"] = SHE(model.features, model.fc)
detectors["MSP"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["GEN"] = GEN(model)
detectors["MaxLogit"] = MaxLogit(model)
detectors["ODIN"] = ODIN(model, norm_std=norm_std, eps=0.002)
detectors["DICE"] = DICE(model=model.features, w=model.fc.weight, b=model.fc.bias, p=0.65)
detectors["RMD"] = RMD(model.features)

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

detectors["NAC-UE"] = NACUE(
    model=model,
    layers=[model.block2, model.block3, model.bn1],
    m_bins=[200, 200, 200],
    alpha=[150.0, 200.0, 250.0],
    o_star=[25, 50, 100],
    device=device,
)

# %%
# Stage 3: Evaluate detectors with benchmark-managed caching
print(f"STAGE 3: Evaluating {len(detectors)} detectors with benchmark-managed caching.")
results = []
benchmark = CIFAR10_OpenOOD(root="data", transform=trans)

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
    for result in res:
        result.update({"Detector": detector_name})
    results += res

df = pd.DataFrame(results)
mean_scores = (
    df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
)
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
