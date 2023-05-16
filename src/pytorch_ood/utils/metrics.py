"""
..  autoclass:: pytorch_ood.utils.OODMetrics
    :members:

"""
from typing import Dict, TypeVar

import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc

from .utils import TensorBuffer, is_unknown

Self = TypeVar("Self")


def calibration_error(
    confidence: torch.Tensor, correct: torch.Tensor, p: str = "2", beta: int = 100
) -> float:
    """
    :see Implementation: `GitHub <https://github.com/hendrycks/natural-adv-examples/>`__

    :param confidence: predicted confidence
    :param correct: ground truth
    :param p: p for norm. Can be one of ``1``, ``2``, or ``infty``
    :param beta: target bin size
    :return: calculated calibration error
    """

    confidence = confidence.numpy()
    correct = correct.numpy()

    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p == "infty" or p == "infinity" or p == "max":
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == "2":
        cerr = np.sqrt(cerr)

    return float(cerr)


def aurra(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    """
    :see Implementation: `GitHub <https://github.com/hendrycks/natural-adv-examples/>`__

    :param confidence: predicted confidence values
    :param correct: ground truth

    :return: score
    """
    conf_ranks = np.argsort(confidence.numpy())[::-1]  # indices from greatest to least confidence
    rra_curve = np.cumsum(np.asarray(correct.numpy())[conf_ranks])
    rra_curve = rra_curve / np.arange(1, len(rra_curve) + 1)  # accuracy at each response rate
    return float(np.mean(rra_curve))


def fpr_at_tpr(pred, target, k=0.95):
    """
    Calculate the False Positive Rate at a certain True Positive Rate

    :param pred: outlier scores
    :param target: target label
    :param k: cutoff value
    :return:
    """
    # results will be sorted in reverse order
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, k)
    if idx == fpr.shape[0]:
        return fpr[idx - 1]

    return fpr[idx]


class OODMetrics(object):
    """
    Calculates various metrics used in OOD detection experiments.

    - AUROC
    - AUPR IN
    - AUPR OUT
    - FPR\\@95TPR

    The interface is similar to ``torchmetrics``.

    .. code :: python

        metrics = OODMetrics()
        outlier_scores = torch.Tensor([0.5, 1.0, -10])
        labels = torch.Tensor([1,2,-1])
        metrics.update(outlier_scores, labels)
        metric_dict = metrics.compute()

    In ``classification`` mode, the inputs will be flattened, so we treat each value as an individual text example.
    Using this mode for segmentation tasks can require a lot of memory and compute.

    In ``segmentation`` mode, the inputs will be flattened along the first (batch) dimension so that the shape is
    :math:`B \\times D` afterwards.
    The scores will then be calculated for each sample in the batch (i.e., over :math:`D` values each), and the final
    score will be the mean over all :math:`B` samples.
    """

    def __init__(self, device: str = "cpu", mode: str = "classification"):
        """
        :param device: where tensors should be stored
        :param mode: either ``classification`` or ``segmentation``.
        """
        super(OODMetrics, self).__init__()
        self.device = device
        self.buffer = TensorBuffer(device=self.device)

        if mode not in ["segmentation", "classification"]:
            raise ValueError("mode must be 'segmentation' or 'classification'")

        self.mode = mode

    def update(self: Self, scores: Tensor, y: Tensor) -> Self:
        """
        Add batch of results to collection.

        :param scores: outlier score
        :param y: target label
        """
        label = is_unknown(y).detach().to(self.device).long()
        scores = scores.detach().to(self.device)

        if self.mode == "classification":
            self.buffer.append("scores", scores)
            self.buffer.append("labels", label)

        elif self.mode == "segmentation":
            # loop along batch dimension
            for i in range(scores.shape[0]):
                metrics = self._compute(label[i].view(-1), scores[i].view(-1))
                for key, value in metrics.items():
                    self.buffer.append(key, value.view(1, -1))

        return self

    def _compute(self, labels: Tensor, scores: Tensor) -> Dict[str, Tensor]:
        """ """
        if len(torch.unique(labels)) != 2:
            raise ValueError("Data must contain IN and OOD samples.")

        scores, scores_idx = torch.sort(scores, stable=True)
        labels = labels[scores_idx]

        auroc = binary_auroc(scores, labels)

        # num_classes=None for binary
        p, r, t = binary_precision_recall_curve(scores, labels)
        aupr_in = auc(r, p)

        p, r, t = binary_precision_recall_curve(-scores, 1 - labels)
        aupr_out = auc(r, p)

        fpr = fpr_at_tpr(scores, labels)

        return {
            "AUROC": auroc,
            "AUPR-IN": aupr_in,
            "AUPR-OUT": aupr_out,
            "FPR95TPR": fpr,
        }

    def compute(self) -> Dict[str, float]:
        """
        Calculate metrics

        :return: dictionary with different metrics
        :raise: ValueError if data does not contain IN and OOD points or buffer is empty
        """
        if self.buffer.is_empty():
            raise ValueError("Must be given data to calculate metrics.")

        if self.mode == "segmentation":
            metrics = {key: self.buffer[key].mean() for key in self.buffer.keys()}

        elif self.mode == "classification":
            labels = self.buffer.get("labels").view(-1)
            scores = self.buffer.get("scores").view(-1)

            if len(torch.unique(labels)) != 2:
                raise ValueError("Data must contain IN and OOD samples.")

            metrics = self._compute(labels, scores)

        metrics = {k: v.item() for k, v in metrics.items()}
        return metrics

    def reset(self: Self) -> Self:
        """
        Resets collected metrics
        """
        self.buffer.clear()
        return self
