"""
..  autoclass:: pytorch_ood.utils.OODMetrics
    :members:

"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

__all__ = [
    "OODMetrics",
    "calibration_error",
    "aurra",
    "fpr_at_tpr",
    "autc_score",
    "oscr_score",
]
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc

from .utils import TensorBuffer, is_unknown


def calibration_error(
    confidence: torch.Tensor, correct: torch.Tensor, p: str = "2", beta: int = 100
) -> float:
    """
    Calibration error of predicted confidences: the samples are sorted by confidence and
    grouped into bins of (about) ``beta`` samples; the error is the :math:`p`-norm of the
    differences between the mean confidence and the accuracy of the bins, weighted by bin size.
    Requires CPU tensors that do not require gradients.

    :see Implementation: `Natural adversarial examples (Hendrycks et al.) on GitHub <https://github.com/hendrycks/natural-adv-examples/>`__

    :param confidence: predicted confidence per sample, of shape :math:`N`
    :param correct: 1 where the prediction was correct, else 0, of shape :math:`N`
    :param p: norm; one of ``"1"``, ``"2"``, or ``"infty"``
    :param beta: target bin size (number of samples per bin)
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
    Area under the risk-response-rate curve (AURRA): the mean accuracy over all response rates,
    when samples are answered in order of decreasing confidence.
    Requires CPU tensors that do not require gradients.

    :see Implementation: `Natural adversarial examples (Hendrycks et al.) on GitHub <https://github.com/hendrycks/natural-adv-examples/>`__

    :param confidence: predicted confidence values, of shape :math:`N`
    :param correct: 1 where the prediction was correct, else 0, of shape :math:`N`

    :return: score
    """
    conf_ranks = np.argsort(confidence.numpy())[::-1]  # indices from greatest to least confidence
    rra_curve = np.cumsum(np.asarray(correct.numpy())[conf_ranks])
    rra_curve = rra_curve / np.arange(1, len(rra_curve) + 1)  # accuracy at each response rate
    return float(np.mean(rra_curve))


def fpr_at_tpr(pred, target, k=0.95):
    """
    Calculate the False Positive Rate at a certain True Positive Rate

    :param pred: outlier scores of shape :math:`N`
    :param target: binary labels of shape :math:`N`, 1 for OOD and 0 for ID
    :param k: target true positive rate, as a fraction in :math:`[0, 1]`
    :return: false positive rate at the first threshold with a true positive rate :math:`\\geq k`
    """
    # results will be sorted in reverse order
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, k)
    if idx == fpr.shape[0]:
        return fpr[idx - 1]

    return fpr[idx]


def autc_score(labels: Tensor, scores: Tensor, pos_label: int = 1) -> Tensor:
    """
    Calculate the Area Under the Threshold Curve (AUTC).

    The AUTC is the mean of the areas under the False Positive Rate and False Negative Rate
    curves, integrated over thresholds :math:`\\tau \\in [0, 1]` after min-max normalizing the
    scores. Lower is better. Using the fact that, for scores normalized to :math:`[0, 1]`,

    .. math::

        \\int_0^1 \\mathrm{FPR}(\\tau)\\,d\\tau = \\frac{1}{N_-}\\sum_{i:\\,y_i=0} s_i,
        \\qquad
        \\int_0^1 \\mathrm{FNR}(\\tau)\\,d\\tau = \\frac{1}{N_+}\\sum_{i:\\,y_i=1} (1 - s_i),

    the score is computed in closed form in :math:`O(N)`, which is exact and avoids
    materializing a threshold curve.

    Here, :math:`s_i` is the min-max normalized score of sample :math:`i`, :math:`y_i = 1` for
    positive (OOD) and :math:`y_i = 0` for negative (ID) samples, and :math:`N_+` and :math:`N_-`
    are the numbers of positive and negative samples. Note that, other than most functions in
    this library, this function expects binary labels and not the labels :math:`< 0` for OOD.
    The scores must not be constant.

    :param labels: ground truth labels, where ``pos_label`` denotes the positive (OOD) class
    :param scores: predicted outlier scores for each sample (higher = more likely positive)
    :param pos_label: value in ``labels`` that denotes the positive class
    :return: AUTC score, lower is better

    :see Paper: `Paper introducing the AUTC (arXiv 2306.14658) <https://arxiv.org/pdf/2306.14658>`__
    """
    labels = labels == pos_label

    # all scores must be between 0 and 1
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    aufpr = scores[~labels].mean()  # area under FPR(tau) over tau in [0, 1]
    aufnr = (1 - scores[labels]).mean()  # area under FNR(tau) over tau in [0, 1]

    return (aufpr + aufnr) / 2


class OODMetrics(object):
    """
    Calculates various metrics used in OOD detection experiments. OOD samples have labels
    :math:`< 0`; they are the positive class, and larger outlier scores mean more likely OOD.
    :meth:`compute` returns a dictionary with the keys

    - ``AUROC``: area under the ROC curve (see the
      `Baseline for Detecting Misclassified and Out-of-Distribution Examples <https://arxiv.org/pdf/1610.02136>`__
      or the `ODIN paper <https://arxiv.org/pdf/1706.02690>`__ for more information)
    - ``AUTC``: area under the threshold curve, lower is better (see :func:`~pytorch_ood.utils.metrics.autc_score` and the
      `AUTC paper <https://arxiv.org/pdf/2306.14658>`__)
    - ``AUPR-IN``: area under the precision-recall curve with ID samples as the positive class
      (see the `Baseline <https://arxiv.org/pdf/1610.02136>`__ or the
      `ODIN paper <https://arxiv.org/pdf/1706.02690>`__)
    - ``AUPR-OUT``: area under the precision-recall curve with OOD samples as the positive class
      (see the `Baseline <https://arxiv.org/pdf/1610.02136>`__ or the
      `ODIN paper <https://arxiv.org/pdf/1706.02690>`__)
    - ``FPR95TPR``: false positive rate at a true positive rate of 95% (see the
      `ODIN paper <https://arxiv.org/pdf/1706.02690>`__)
    - ``ACC``: closed-set classification accuracy on the known (in-distribution) samples,
      included automatically whenever predicted class indices are passed to
      :meth:`update`.

    The interface is similar to ``torchmetrics``.

    .. code-block:: python

        import torch

        from pytorch_ood.utils import OODMetrics

        metrics = OODMetrics()
        outlier_scores = torch.tensor([0.5, 1.0, -10.0])
        labels = torch.tensor([1, 2, -1])
        metrics.update(outlier_scores, labels)
        metric_dict = metrics.compute()

    Passing predicted class indices additionally reports closed-set accuracy
    (``model``, ``detector``, ``x`` and ``labels`` are a classifier, a detector, a batch of
    inputs and its labels):

    .. code-block:: python

        metrics = OODMetrics()
        logits = model(x)
        outlier_scores = detector(x)
        metrics.update(outlier_scores, labels, logits.argmax(dim=1))
        metric_dict = metrics.compute()  # now also contains "ACC"

    In ``classification`` mode, the inputs will be flattened, so we treat each value as an individual example.
    Using this mode for segmentation tasks can require a lot of memory and compute.

    In ``segmentation`` mode, scores and labels must have the shape :math:`B \\times H \\times W`.
    The metrics are calculated for each of the :math:`B` samples in the batch separately (over its
    :math:`H \\cdot W` pixels), and the final result is the mean over all samples. Each sample must
    therefore contain both ID and OOD pixels.
    """

    def __init__(
        self, device: str = "cpu", mode: str = "classification", void_label: Optional[int] = None
    ):
        """
        :param device: where tensors should be stored
        :param mode: either ``classification`` or ``segmentation``.
        :param void_label: label that will be ignored during score calculation
        :raises ValueError: if ``mode`` is invalid
        """
        super(OODMetrics, self).__init__()
        self.device = device
        self.buffer = TensorBuffer(device=device)
        self.void_label = void_label

        if mode not in ["segmentation", "classification"]:
            raise ValueError("mode must be 'segmentation' or 'classification'")

        self.mode = mode

    def update(self, scores: Tensor, y: Tensor, predictions: Optional[Tensor] = None) -> Self:
        """
        Add batch of results to collection.

        :param scores: outlier scores, of shape :math:`B` (``classification``) or
            :math:`B \\times H \\times W` (``segmentation``). Larger means more likely OOD.
        :param y: target labels of the same shape as ``scores``; values :math:`< 0` denote OOD
        :param predictions: predicted class indices of the same shape as ``y``, classification
            mode only. When given (on every call to this instance), :meth:`compute` additionally
            reports closed-set accuracy ("ACC") on the known (in-distribution) samples.
        :return: self
        :raises ValueError: if the shapes of the inputs do not match
        :raises NotImplementedError: if ``predictions`` are given in segmentation mode
        """
        scores = scores.detach()
        y = y.detach()

        if y.shape != scores.shape:
            raise ValueError(f"Inputs have wrong size: {y.shape} and {scores.shape}")

        if self.mode == "classification":
            self.buffer.append("scores", scores)
            self.buffer.append("y", y)

            if predictions is not None:
                predictions = predictions.detach()
                if predictions.shape != y.shape:
                    raise ValueError(f"Inputs have wrong size: {predictions.shape} and {y.shape}")
                self.buffer.append("predictions", predictions)

        elif self.mode == "segmentation":
            if predictions is not None:
                raise NotImplementedError(
                    "predictions/accuracy are not supported in segmentation mode"
                )

            # Should contain BxHxW
            assert len(scores.shape) == 3
            assert len(y.shape) == 3

            assert scores.device == y.device, "Score and target tensor must be on same device"

            # loop along batch dimension
            for i in range(scores.shape[0]):
                # computation will be carried out on the device where the data currently resides
                # since this is usually a gpu, this speeds up the processing drastically,
                # since only the reduced results have to be stored.
                metrics = self._compute(y[i].view(-1), scores[i].view(-1))
                for key, value in metrics.items():
                    self.buffer.append(key, value.view(1, -1))

        return self

    @torch.no_grad()
    def _compute(self, labels: Tensor, scores: Tensor) -> Dict[str, Tensor]:
        """ """
        if labels.shape != scores.shape:
            raise ValueError(f"Inputs have wrong size: {labels.shape} and {scores.shape}")

        # filter all void labels
        if self.void_label is not None:
            void_mask = labels != self.void_label
            labels = labels[void_mask]
            scores = scores[void_mask]

        # map OOD to 1 (positive), map ID to 0 (negative)
        labels = is_unknown(labels).long()

        # there must now be ID and OOD samples
        if len(torch.unique(labels)) != 2:
            raise ValueError("Data must contain ID and OOD samples.")

        scores, scores_idx = torch.sort(scores, stable=True)
        labels = labels[scores_idx]

        auroc = binary_auroc(scores, labels)

        autc = autc_score(labels, scores, pos_label=1)

        # num_classes=None for binary
        p, r, t = binary_precision_recall_curve(scores, labels)
        aupr_out = auc(r, p)

        p, r, t = binary_precision_recall_curve(-scores, 1 - labels)
        aupr_in = auc(r, p)

        fpr = fpr_at_tpr(scores, labels)

        return {
            "AUROC": auroc.cpu(),
            "AUTC": autc.cpu(),
            "AUPR-IN": aupr_in.cpu(),
            "AUPR-OUT": aupr_out.cpu(),
            "FPR95TPR": fpr.cpu(),
        }

    def compute(self) -> Dict[str, float]:
        """
        Calculate metrics

        :return: dictionary with the keys ``AUROC``, ``AUTC``, ``AUPR-IN``, ``AUPR-OUT`` and
            ``FPR95TPR`` (and ``ACC``, if predictions were given)
        :raises ValueError: if the buffer is empty or the data does not contain both ID and OOD points
        """
        if self.buffer.is_empty():
            raise ValueError("Must be given data to calculate metrics.")

        if self.mode == "segmentation":
            metrics = {key: self.buffer[key].mean() for key in self.buffer.keys()}

        elif self.mode == "classification":
            labels = self.buffer.get("y").view(-1)
            scores = self.buffer.get("scores").view(-1)

            metrics = self._compute(labels, scores)

            if "predictions" in self.buffer:
                predictions = self.buffer.get("predictions").view(-1)

                # mirror the void-label filtering _compute() applies internally
                if self.void_label is not None:
                    void_mask = labels != self.void_label
                    labels = labels[void_mask]
                    predictions = predictions[void_mask]

                known = labels >= 0
                if known.any():
                    metrics["ACC"] = (predictions[known] == labels[known]).float().mean().cpu()

        metrics = {k: v.item() for k, v in metrics.items()}
        return metrics

    def reset(self) -> Self:
        """
        Resets collected metrics
        """
        self.buffer.clear()
        return self


@torch.no_grad()
def oscr_score(outlier_scores: Tensor, predictions: Tensor, labels: Tensor) -> float:
    """
    Open-Set Classification Rate (OSCR): the area under the OSCR curve, which measures closed-set
    classification and open-set detection jointly.

    A sample is accepted as known if its outlier score :math:`s(x)` is at most a threshold
    :math:`\\tau`. The curve plots the Correct Classification Rate, the fraction of known samples
    :math:`\\mathcal{D}_c` that are accepted and correctly classified,

    .. math::
        \\mathrm{CCR}(\\tau) = \\frac{|\\{x \\in \\mathcal{D}_c : \\hat{y}(x) = y(x) \\wedge s(x) \\leq \\tau\\}|}
        {|\\mathcal{D}_c|}

    against the False Positive Rate, the fraction of unknown samples :math:`\\mathcal{D}_u` that are
    accepted,

    .. math::
        \\mathrm{FPR}(\\tau) = \\frac{|\\{x \\in \\mathcal{D}_u : s(x) \\leq \\tau\\}|}{|\\mathcal{D}_u|}

    for all thresholds :math:`\\tau`. The OSCR is at most the closed-set accuracy, which a perfect
    detector reaches. A random detector gives about half of the closed-set accuracy.

    .. rubric:: Examples

    .. code-block:: python

        scores = detector(x)
        preds = model(x).argmax(dim=1)
        result = oscr_score(scores, preds, labels)

    :param outlier_scores: outlier scores :math:`s(x)`, shape :math:`B`
    :param predictions: predicted classes :math:`\\hat{y}(x)`, shape :math:`B`
    :param labels: labels :math:`y(x)`, shape :math:`B`; labels :math:`< 0` mark unknown samples
    :return: OSCR in :math:`[0, 1]`
    :raises ValueError: if ``labels`` contain no known or no unknown samples

    :see Paper: `Reducing Network Agnostophobia <https://arxiv.org/abs/1811.04110>`__
    """
    known_mask = labels >= 0

    s_id = outlier_scores[known_mask].cpu()
    s_ood = outlier_scores[~known_mask].cpu()
    correct = (predictions[known_mask] == labels[known_mask]).cpu()

    n_id = s_id.shape[0]
    n_ood = s_ood.shape[0]

    if n_id == 0 or n_ood == 0:
        raise ValueError("oscr_score requires both known and unknown samples.")

    # Every unique score is a threshold, so the curve is a step function. Where known and
    # unknown samples tie, CCR and FPR change at the same threshold and the curve has a diagonal
    # segment, as for the ROC curve. The paper counts CCR with s < tau and FPR with s <= tau
    # instead, which makes ties pessimistic (constant scores would give 0, not accuracy / 2).
    thresholds = torch.unique(torch.cat([s_id, s_ood]))

    sort_idx = torch.argsort(s_id)
    correct_counts = torch.cat([torch.zeros(1), torch.cumsum(correct[sort_idx].float(), dim=0)])
    accepted_id = torch.searchsorted(s_id[sort_idx], thresholds, right=True)
    ccr = correct_counts[accepted_id] / n_id

    s_ood_sorted, _ = torch.sort(s_ood)
    fpr = torch.searchsorted(s_ood_sorted, thresholds, right=True).float() / n_ood

    # starts at (0, 0); the largest threshold accepts everything, so the curve ends at (1, accuracy)
    fpr = torch.cat([torch.zeros(1), fpr])
    ccr = torch.cat([torch.zeros(1), ccr])

    return float(torch.trapezoid(ccr, fpr).item())
