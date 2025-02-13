"""
Newsgroups Outlier Exposure
==============================

Benchmark code for Newsgroups 20, trained with Outlier Exposure on the WikiText2 dataset.
We test the detectors against three different Text dataset
and calculate the mean performance.

Uses GRU model from the OOD detection baseline paper.

The original results can not be reproduced, as the dictionaries (word-to-token-mappings) are not available.

+-------------+-------+---------+----------+----------+
| Detector    | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=========+==========+==========+
| ViM         | 57.87 | 67.14   | 65.42    | 53.18    |
+-------------+-------+---------+----------+----------+
| Mahalanobis | 63.27 | 68.39   | 69.40    | 50.52    |
+-------------+-------+---------+----------+----------+
| KLMatching  | 92.92 | 91.70   | 93.17    | 21.32    |
+-------------+-------+---------+----------+----------+
| MaxSoftmax  | 93.55 | 92.27   | 94.50    | 20.17    |
+-------------+-------+---------+----------+----------+
| Entropy     | 94.33 | 93.05   | 95.10    | 19.76    |
+-------------+-------+---------+----------+----------+
| EnergyBased | 94.63 | 93.14   | 95.67    | 16.90    |
+-------------+-------+---------+----------+----------+
| MaxLogit    | 94.66 | 93.14   | 95.66    | 17.18    |
+-------------+-------+---------+----------+----------+


"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from pytorch_ood.dataset.txt import (
    Multi30k,
    NewsGroup20,
    Reuters52,
    WikiText2,
    WMT16Sentences,
)
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
from pytorch_ood.loss import OutlierExposureLoss
from pytorch_ood.model import GRUClassifier
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, is_known

fix_random_seed(123)

n_epochs = 5
lr = 0.001
device = "cuda:0"
root = "data"

# %%
# download datasets
train_dataset_in = NewsGroup20(root, train=True, download=True)

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_dataset_in))
vocab.set_default_index(0)


def prep(x):
    return torch.tensor([vocab[v] for v in tokenizer(x)], dtype=torch.int64)


# %%
train_dataset_in = NewsGroup20(root, train=True, transform=prep)
dataset_in_test = NewsGroup20(root, train=False, transform=prep)
train_ood_dataset = WikiText2(
    root, split="train", download=True, transform=prep, target_transform=ToUnknown()
)

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


loader_train = DataLoader(
    train_dataset_in + train_ood_dataset,
    batch_size=20,
    shuffle=True,
    collate_fn=collate_batch,
)
loader_in_test = DataLoader(dataset_in_test, batch_size=16, shuffle=True, collate_fn=collate_batch)

# %% Create a neural network
print("STAGE 1: Train Model")
model = GRUClassifier(num_classes=20, n_vocab=len(vocab))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
criterion = OutlierExposureLoss()

model.to(device)

# %% Train model
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")

    model.train()
    loss_ema = None
    correct = 0
    total = 0

    bar = tqdm(loader_train)

    for n, batch in enumerate(bar):
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ema = loss.item() if not loss_ema else loss_ema * 0.99 + loss.item() * 0.01

        pred = logits.max(dim=1).indices
        correct += pred[is_known(labels)].eq(labels[is_known(labels)]).sum().data.cpu().item()
        total += is_known(labels).sum()

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
detectors["Mahalanobis"] = Mahalanobis(model.features, eps=0.0)
detectors["KLMatching"] = KLMatching(model)
detectors["MaxSoftmax"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)


print(f"> Fitting {len(detectors)} detectors")

for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.fit(loader_train, device=device)

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


# %% calculate mean scores over all datasets, use percent

df = pd.DataFrame(results)
mean_scores = df.groupby("Detector").mean() * 100
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
