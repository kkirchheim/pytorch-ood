"""

CIFAR 10 Cached Baseline
========================

Benchmark code for CIFAR-10 that reuses cached logits and pooled features
for fitting detectors that expose the standard representation APIs. At
evaluation time, cached logits are used for true logits detectors, while
feature detectors that operate directly on pooled features can use the
cached feature representation. Other detectors still use their regular
``predict()`` interface so their full model-time behavior is preserved.

"""

from copy import deepcopy
import gc
import pandas as pd  # additional dependency, used here for convenience
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST
from tqdm.auto import tqdm  # additional dependency, used here for convenience

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Places365,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
)
from pytorch_ood.api import FeaturesDetector, LogitsDetector
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
from pytorch_ood.utils import (
    OODMetrics,
    TensorBuffer,
    ToUnknown,
    extract_features,
    fix_random_seed,
)

device = "cuda:0"

fix_random_seed(123)


def cache_representations(
    data_loader: DataLoader, model: nn.Module, feature_model: nn.Module, device: str
):
    """
    Cache logits, pooled features, and labels for all samples in a loader.
    Unlike ``extract_features(...)``, this keeps OOD samples and labels.
    """
    print(f"Extracting cached logits/features from {len(data_loader.dataset)} samples")
    buffer = TensorBuffer()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            buffer.append("logits", model(x).view(x.shape[0], -1))
            buffer.append("features", feature_model(x).view(x.shape[0], -1))
            buffer.append("label", y)

    return {
        "logits": buffer.get("logits"),
        "features": buffer.get("features"),
        "labels": buffer.get("label"),
    }


def fit_detector(
    detector, train_loader: DataLoader, train_cache: dict, train_labels: torch.Tensor, device: str
):
    """
    Fit using cached representations for detectors with a standard semantic
    interface. Otherwise fall back to ``fit(...)``.
    """
    if detector.requires_fit and isinstance(detector, LogitsDetector):
        detector.fit_logits(train_cache["logits"], train_labels)
        return

    if detector.requires_fit and isinstance(detector, FeaturesDetector):
        detector.fit_features(train_cache["features"], train_labels)
        return

    if detector.requires_fit:
        detector.to(device)
        detector.fit(train_loader)


def evaluate_detector(detector, data_loader: DataLoader, eval_cache: dict, device: str) -> dict:
    """
    Evaluate using cached logits for logits detectors and cached pooled
    features for feature detectors whose ``predict(x)`` does not add extra
    input preprocessing. Otherwise fall back to ``predict(x)``.
    """
    metrics = OODMetrics()
    detector.to(device)

    if isinstance(detector, LogitsDetector):
        scores = detector.predict_logits(eval_cache["logits"])
        metrics.update(scores, eval_cache["labels"].to(scores.device))
        return metrics.compute()

    if isinstance(detector, FeaturesDetector):
        if isinstance(detector, Mahalanobis) and detector.eps > 0:
            for x, y in tqdm(data_loader, leave=False):
                metrics.update(detector(x.to(device)), y.to(device))
        else:
            scores = detector.predict_features(eval_cache["features"])
            metrics.update(scores, eval_cache["labels"].to(scores.device))

        return metrics.compute()

    for x, y in tqdm(data_loader, leave=False):
        metrics.update(detector(x.to(device)), y.to(device))

    return metrics.compute()


# %%
# Setup preprocessing
trans = WideResNet.transform_for("cifar10-pt")
norm_std = WideResNet.norm_std_for("cifar10-pt")

# %%
# Setup datasets

dataset_in_test = CIFAR10(root="data", train=False, transform=trans, download=True)

ood_datasets = [
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    LSUNCrop,
    LSUNResize,
    Places365,
    CIFAR100,
    MNIST,
    FashionMNIST,
]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128, num_workers=12)
    datasets[ood_dataset.__name__] = test_loader

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
# Stage 3: Fit detectors
print(f"STAGE 3: Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(
    CIFAR10(root="data", train=True, transform=trans), batch_size=128, num_workers=12
)

print("Extracting training logits")
train_logits, train_labels_logits = extract_features(loader_in_train, model, device)
print("Extracting training pooled features")
train_features, train_labels_features = extract_features(loader_in_train, model.features, device)

assert torch.equal(train_labels_logits, train_labels_features)
train_labels = train_labels_logits
train_cache = {
    "logits": train_logits,
    "features": train_features,
}

for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    fit_detector(detector, loader_in_train, train_cache, train_labels, device)

# %%
# Stage 4: Evaluate detectors
print(f"STAGE 4: Evaluating {len(detectors)} detectors on {len(datasets)} datasets.")
results = []

for dataset_name, loader in datasets.items():
    print(f"> Caching representations for {dataset_name}")
    eval_cache = cache_representations(loader, model, model.features, device)

    for detector_name, detector in detectors.items():
        print(f"--> {detector_name}")
        scores = evaluate_detector(detector, loader, eval_cache, device)
        result = {"Detector": detector_name, "Dataset": dataset_name}
        result.update(scores)
        results.append(result)

    del eval_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df = pd.DataFrame(results)
mean_scores = (
    df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
)
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
