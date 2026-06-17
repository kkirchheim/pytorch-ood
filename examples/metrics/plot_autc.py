"""
AUTC
-------------------------

Historgram and Metrics for random scores with different delta.

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional.classification import binary_roc

from pytorch_ood.utils import OODMetrics


def fpr_tpr_curve(labels, scores, pos_label=1):
    """Compute FPR/TPR over thresholds for min-max normalized scores."""
    labels = (labels == pos_label).long()
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    fpr, tpr, thresholds = binary_roc(scores, labels)
    # prepend the (threshold=1) point where nothing is flagged as positive
    fpr = torch.cat([torch.tensor([0.0], device=fpr.device), fpr])
    tpr = torch.cat([torch.tensor([0.0], device=tpr.device), tpr])
    thresholds = torch.cat([torch.tensor([1.0], device=thresholds.device), thresholds])
    return fpr, tpr, thresholds

# %%
# Parameters

# delta between in and ood data
near_delta = 2
far_delta = 10

# split
in_samples_num = 9
out_samples_num = 1

# random torch tensors
offset = 10**3
in_scores = torch.rand(in_samples_num * offset)
out_scores = torch.rand(out_samples_num * offset)

# %%
# Define function


def metrics_and_plots(in_scores, out_scores, delta, name):
    metrics = OODMetrics()
    # concat all scores
    scores = torch.cat([in_scores, out_scores + delta])
    # create labels
    labels = torch.cat([torch.zeros_like(in_scores), torch.ones_like(out_scores)])
    metrics.update(scores, -labels)
    metric_dict = metrics.compute()
    print(name, metric_dict)

    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram
    axes[0].hist(
        in_scores.cpu().numpy(),
        bins=100,
        alpha=0.5,
        label="In-Distribution",
        color="tab:blue",
    )
    axes[0].hist(
        (out_scores + delta).cpu().numpy(),
        bins=100,
        alpha=0.5,
        label="Out-of-Distribution",
        color="tab:orange",
    )
    axes[0].set_title(f"{name} Histogram", weight="bold")
    axes[0].set_xlabel("Scores")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(loc="upper right")

    # Plot FPR and FNR curve
    fpr, tpr, thresholds = fpr_tpr_curve(labels, scores)
    axes[1].plot(thresholds, fpr, label="FPR", color="tab:blue")
    axes[1].plot(thresholds, 1 - tpr, label="FNR", color="tab:orange")
    axes[1].set_title(f"{name} FPR and FNR", weight="bold")
    axes[1].set_xlabel("Thresholds")
    axes[1].set_ylabel("Rate")
    axes[1].legend(loc="best")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{name}_metrics_plots.png")
    plt.show()


# %%
# Plot and calculate metrics
metrics_and_plots(in_scores, out_scores, near_delta, "Near")
metrics_and_plots(in_scores, out_scores, far_delta, "Far")
