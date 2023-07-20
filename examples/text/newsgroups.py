"""
Newsgroups 20
==============================

Benchmark code for Newsgroups 20. We test the models against three different Text dataset
and calculate the mean performance.

Uses GRU model from the OOD detection baseline paper.

The original results can not be reproduced, as the dictionaries (word-to-token-mappings) are not available.

+-------------+-------+---------+----------+----------+
| Detector    | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=========+==========+==========+
| KLMatching  | 80.17 | 84.72   | 70.21    | 54.91    |
+-------------+-------+---------+----------+----------+
| MaxSoftmax  | 80.26 | 84.67   | 72.13    | 53.39    |
+-------------+-------+---------+----------+----------+
| Entropy     | 84.14 | 87.46   | 75.37    | 51.67    |
+-------------+-------+---------+----------+----------+
| ViM         | 87.00 | 88.94   | 81.77    | 39.80    |
+-------------+-------+---------+----------+----------+
| MaxLogit    | 88.32 | 89.59   | 81.95    | 39.24    |
+-------------+-------+---------+----------+----------+
| EnergyBased | 88.82 | 89.85   | 82.61    | 37.96    |
+-------------+-------+---------+----------+----------+
| Mahalanobis | 89.06 | 89.40   | 85.56    | 35.24    |
+-------------+-------+---------+----------+----------+

"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from pytorch_ood.dataset.txt import Multi30k, NewsGroup20, Reuters52, WMT16Sentences
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
)
from pytorch_ood.model import GRUClassifier
from pytorch_ood.utils import OODMetrics, ToUnknown

torch.manual_seed(123)

n_epochs = 10
lr = 0.001
device = "cuda:0"

# %%


# download datasets
train_dataset = NewsGroup20("data/", train=True, download=True)

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_dataset))


def prep(x):
    return torch.tensor([vocab[v] for v in tokenizer(x)], dtype=torch.int64)


# %%
train_dataset = NewsGroup20("data/", train=True, transform=prep)
dataset_in_test = NewsGroup20("data/", train=False, transform=prep)

# %%
# Add padding, etc.


def collate_batch(batch):
    texts = [i[0] for i in batch]
    labels = torch.tensor([i[1] for i in batch], dtype=torch.int64)
    t_lengths = torch.tensor([len(t) for t in texts])
    max_t_length = torch.max(t_lengths)

    padded = []
    for text in texts:
        t = torch.cat([torch.zeros(max_t_length - len(text), dtype=torch.long), text])
        padded.append(t)
    return torch.stack(padded, dim=0), labels


loader_in_train = DataLoader(
    train_dataset, batch_size=20, shuffle=True, collate_fn=collate_batch
)
loader_in_test = DataLoader(
    dataset_in_test, batch_size=16, shuffle=True, collate_fn=collate_batch
)

# %% Create a neural network
print("STAGE 1: Train Model")
model = GRUClassifier(num_classes=20, n_vocab=len(vocab))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)


# %% Train model
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")

    model.train()
    loss_ema = 0
    correct = 0
    total = 0

    model.train()
    for n, batch in enumerate(loader_in_train):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ema = loss_ema * 0.9 + loss.data.cpu().item() * 0.1

        pred = logits.max(dim=1).indices
        correct += pred.eq(labels).sum().data.cpu().item()
        total += pred.shape[0]

        if n % 10 == 0:
            print(f"{loss_ema:.2f} {correct / total:.2%}")

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
detectors["Mahalanobis"] = Mahalanobis(model.features, eps=0.0)
detectors["KLMatching"] = KLMatching(model)
detectors["MaxSoftmax"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)


print(f"> Fitting {len(detectors)} detectors")

for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.fit(loader_in_train, device=device)

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
mean_scores = df.groupby("Detector").mean() * 100
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
