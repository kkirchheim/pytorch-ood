"""
Newsgroups 20
==============================

Benchmark code for Newsgroups 20. We test the models against three different Text dataset
and calculate the mean performance.

Uses GRU model from the OOD detection baseline paper.

The original results can not be reproduced, as the dictionaries (word-to-token-mappings) are not available.

+-------------+-------+-------+---------+----------+----------+
| Detector    | AUROC | AUTC  | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=======+=========+==========+==========+
| MaxSoftmax  | 71.47 | 41.83 | 61.00   | 79.01    | 68.68    |
+-------------+-------+-------+---------+----------+----------+
| KLMatching  | 72.64 | 42.98 | 55.75   | 80.66    | 79.35    |
+-------------+-------+-------+---------+----------+----------+
| Entropy     | 73.58 | 41.64 | 62.25   | 80.77    | 68.19    |
+-------------+-------+-------+---------+----------+----------+
| MaxLogit    | 79.50 | 41.37 | 70.57   | 83.71    | 58.08    |
+-------------+-------+-------+---------+----------+----------+
| EnergyBased | 80.07 | 41.07 | 71.32   | 83.94    | 57.39    |
+-------------+-------+-------+---------+----------+----------+
| ViM         | 84.37 | 40.81 | 78.93   | 86.10    | 46.19    |
+-------------+-------+-------+---------+----------+----------+
| Mahalanobis | 86.84 | 39.27 | 82.65   | 87.45    | 42.04    |
+-------------+-------+-------+---------+----------+----------+


"""

# sphinx_gallery_thumbnail_path = "_static/thumbs/text.png"

import re
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_ood.dataset.txt import Multi30k, NewsGroup20, Reuters52, WMT16Sentences
from pytorch_ood.detector import (
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
)
from pytorch_ood.metrics import OODMetrics
from pytorch_ood.model import GRUClassifier
from pytorch_ood.utils import ToUnknown, fix_random_seed

fix_random_seed(123)

n_epochs = 10
lr = 0.001
device = "cuda:0"
root = "data"

# %%


# download datasets
train_dataset = NewsGroup20(root, train=True, download=True)

# index 0 is for unknown tokens, index 1 for padding (the GRUClassifier embeds it as zero)
UNK, PAD = 0, 1


def tokenize(text):
    """Lowercase, then split into words and punctuation marks."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


# vocabulary of all tokens in the training set, most frequent first
counts = Counter(token for text, _ in train_dataset for token in tokenize(text))
vocab = {token: i for i, (token, _) in enumerate(counts.most_common(), start=2)}


def prep(x):
    return torch.tensor([vocab.get(t, UNK) for t in tokenize(x)], dtype=torch.int64)


# %%
train_dataset = NewsGroup20(root, train=True, transform=prep, download=True)
dataset_in_test = NewsGroup20(root, train=False, transform=prep, download=True)

# %%
# Add padding, etc.


def collate_batch(batch):
    texts = [i[0] for i in batch]
    labels = torch.tensor([i[1] for i in batch], dtype=torch.int64)
    t_lengths = torch.tensor([len(t) for t in texts])
    max_t_length = torch.max(t_lengths)

    padded = []
    for text in texts:
        t = torch.cat([torch.full((max_t_length - len(text),), PAD, dtype=torch.long), text])
        padded.append(t)
    return torch.stack(padded, dim=0), labels


loader_in_train = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=collate_batch)
loader_in_test = DataLoader(dataset_in_test, batch_size=16, shuffle=True, collate_fn=collate_batch)

# %% Create a neural network
print("STAGE 1: Train Model")
model = GRUClassifier(num_classes=20, n_vocab=len(vocab) + 2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)


# %% Train model
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")

    model.train()
    loss_ema = None
    correct = 0
    total = 0

    model.train()

    bar = tqdm(loader_in_train)

    for n, batch in enumerate(bar):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ema = loss.item() if not loss_ema else loss_ema * 0.99 + loss.item() * 0.01

        pred = logits.max(dim=1).indices
        correct += pred.eq(labels).sum().data.cpu().item()
        total += pred.shape[0]

        bar.set_postfix_str(f"loss: {loss:.2f} acc: {correct / total:.2%}")

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for n, batch in enumerate(loader_in_test):
            inputs, labels = batch

            inputs = inputs.cuda()
            labels = labels.cuda()
            logits = model(inputs)
            pred = logits.max(dim=1).indices
            correct += pred.eq(labels).sum().data.cpu().item()
            total += pred.shape[0]

        print(f"Test Accuracy: {correct / total:.2%}")

# %% Create test datasets

ood_datasets = [Reuters52, Multi30k, WMT16Sentences]

datasets = {}

for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=prep, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(
        dataset_in_test + dataset_out_test, batch_size=16, collate_fn=collate_batch
    )
    datasets[ood_dataset.__name__] = test_loader

# %% Create Detectors
# Fit detectors to training data (some require this, some do not)
print("STAGE 2: Creating OOD Detectors")

detectors = {}
detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis"] = Mahalanobis(model.features)
detectors["KLMatching"] = KLMatching(model)
detectors["MaxSoftmax"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)


print(f"> Fitting {len(detectors)} detectors")

for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.to(device)
    detector.fit(loader_in_train)

# %% Evaluate
print(f"STAGE 3: Evaluating {len(detectors)} detectors on {len(datasets)} datasets.")
results = []

with torch.no_grad():
    for detector_name, detector in detectors.items():
        print(f"> Evaluating {detector_name}")
        for dataset_name, loader in datasets.items():
            print(f"--> {dataset_name}")
            metrics = OODMetrics()
            for x, y in loader:
                metrics.update(detector(x.to(device)), y.to(device))

            r = {"Detector": detector_name, "Dataset": dataset_name}

            r.update(metrics.compute())
            results.append(r)


# calculate mean scores over all datasets, use percent

df = pd.DataFrame(results)
mean_scores = (
    df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean() * 100
)
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
