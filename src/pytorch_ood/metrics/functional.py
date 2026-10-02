"""
Functional metrics: stateless functions on complete tensors, on any device.

The functions follow the label convention of the library: labels :math:`< 0` mark OOD samples,
which are the positive class, and larger outlier scores mean more likely OOD. Unlike
classification metric libraries, they never transform the scores, so scores of any range and
dtype are supported.
"""

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ..utils.utils import _check_fraction

__all__ = [
    "roc_curve",
    "pr_curve",
    "auroc",
    "aupr",
    "fpr_at_tpr",
    "autc",
    "accuracy",
    "oscr_score",
    "calibration_error",
    "aurra",
    "calc_openness",
]


def _float64(x: Tensor) -> Tensor:
    # MPS does not support float64
    if x.device.type == "mps":
        x = x.cpu()
    return x.double()


def _area(x: Tensor, y: Tensor) -> Tensor:
    """Trapezoidal area under the curve through the points ``(x, y)``, in the order given."""
    return ((x[1:] - x[:-1]) * (y[1:] + y[:-1])).sum() / 2


def _flatten(values: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    if values.shape != labels.shape:
        raise ValueError(
            f"Inputs must have the same shape, got {tuple(values.shape)} and {tuple(labels.shape)}"
        )
    return values.detach().reshape(-1), labels.detach().reshape(-1).to(values.device)


def _check_scores(scores: Tensor) -> None:
    if scores.is_floating_point():
        n_nan = int(torch.isnan(scores).sum())
        if n_nan:
            raise ValueError(f"scores contain {n_nan} NaN values")


class _Counts:
    """
    Cumulative counts of positive and negative samples at each unique threshold, the basis of
    all curves. With the thresholds :math:`t_1 > \\dots > t_m` (the unique scores), ``tps[j]``
    and ``fps[j]`` count the positive and negative samples with a score :math:`\\geq t_j`.
    Ties are handled by construction: samples with equal scores enter at the same threshold.
    """

    def __init__(self, scores: Tensor, positive: Tensor):
        _check_scores(scores)
        n = scores.shape[0]
        n_pos = int(positive.sum())
        if n_pos == 0 or n_pos == n:
            raise ValueError(
                "Curves require both positive and negative samples, i.e., OOD (label < 0) and "
                "ID (label >= 0) samples"
            )

        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        # the last index of each group of equal scores; -0.0 == 0.0, so they tie
        last = torch.ones(n, dtype=torch.bool, device=scores.device)
        last[:-1] = sorted_scores[1:] != sorted_scores[:-1]
        idx = torch.nonzero(last).reshape(-1)

        # int64 counts are exact for any number of samples (float32 is exact only up to 2^24)
        self.tps = torch.cumsum(positive[order].long(), dim=0)[idx]
        self.fps = idx + 1 - self.tps
        self.thresholds = sorted_scores[idx]

    @property
    def n_pos(self) -> Tensor:
        return self.tps[-1]

    @property
    def n_neg(self) -> Tensor:
        return self.fps[-1]

    def reversed(self) -> Tuple[Tensor, Tensor]:
        """
        Counts with the roles of positives and negatives swapped and the order of the scores
        reversed, i.e., the counts of ``-scores`` with the negatives as positive class.
        """
        zero = self.tps.new_zeros(1)
        # samples with a score <= t_j are those not counted at t_{j-1}
        tps = self.n_neg - torch.cat([zero, self.fps[:-1]])
        fps = self.n_pos - torch.cat([zero, self.tps[:-1]])
        return tps.flip(0), fps.flip(0)


def _roc(counts: _Counts) -> Tuple[Tensor, Tensor]:
    zero = counts.tps.new_zeros(1)
    fpr = _float64(torch.cat([zero, counts.fps])) / _float64(counts.n_neg)
    tpr = _float64(torch.cat([zero, counts.tps])) / _float64(counts.n_pos)
    return fpr, tpr


def _pr(tps: Tensor, fps: Tensor) -> Tuple[Tensor, Tensor]:
    # as sklearn.metrics.precision_recall_curve: one point per threshold, with decreasing
    # recall, ending at (recall 0, precision 1)
    tps, fps = _float64(tps), _float64(fps)
    precision = tps / (tps + fps)  # every threshold accepts at least one sample
    recall = tps / tps[-1]
    one = precision.new_ones(1)
    return torch.cat([precision.flip(0), one]), torch.cat([recall.flip(0), one - 1])


def _aupr(counts: _Counts, positive: str) -> Tensor:
    if positive == "ood":
        precision, recall = _pr(counts.tps, counts.fps)
    elif positive == "id":
        precision, recall = _pr(*counts.reversed())
    else:
        raise ValueError(f"positive must be 'ood' or 'id', got {positive!r}")
    # recall is decreasing, so the area is negative
    return -_area(recall, precision)


def _fpr_at_tpr(counts: _Counts, tpr: float) -> Tensor:
    fpr_curve, tpr_curve = _roc(counts)
    # first threshold with a true positive rate >= tpr; the curve ends at tpr = 1
    idx = torch.searchsorted(
        tpr_curve, torch.tensor([tpr], dtype=tpr_curve.dtype, device=tpr_curve.device)
    )
    return fpr_curve[idx[0]]


def _counts(scores: Tensor, labels: Tensor) -> _Counts:
    scores, labels = _flatten(scores, labels)
    return _Counts(scores, labels < 0)


def roc_curve(scores: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Receiver operating characteristic, with OOD as the positive class.

    The curve has one point per unique score, starting at :math:`(0, 0)`, where no sample is
    flagged as OOD, and ending at :math:`(1, 1)`.

    :param scores: outlier scores of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :return: false positive rates, true positive rates (both float64), and the thresholds (in
        decreasing order, one fewer than the rates, as the first point has no threshold)
    :raises ValueError: if the shapes differ, the scores contain NaN, or there are no ID or no
        OOD samples
    """
    counts = _counts(scores, labels)
    fpr, tpr = _roc(counts)
    return fpr, tpr, counts.thresholds


def pr_curve(
    scores: Tensor, labels: Tensor, positive: str = "ood"
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Precision-recall curve, as computed by :func:`sklearn.metrics.precision_recall_curve`: one
    point per unique score, with decreasing recall, ending at recall :math:`0` and precision
    :math:`1`.

    :param scores: outlier scores of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :param positive: the positive class, ``"ood"`` or ``"id"``. For ``"id"``, lower scores mean
        more likely positive.
    :return: precisions and recalls (both float64), and the thresholds (in increasing order of
        the outlier score for ``"ood"``, in decreasing order for ``"id"``)
    :raises ValueError: if the shapes differ, the scores contain NaN, there are no ID or no OOD
        samples, or ``positive`` is invalid
    """
    counts = _counts(scores, labels)
    if positive == "ood":
        precision, recall = _pr(counts.tps, counts.fps)
        thresholds = counts.thresholds.flip(0)
    elif positive == "id":
        precision, recall = _pr(*counts.reversed())
        thresholds = counts.thresholds
    else:
        raise ValueError(f"positive must be 'ood' or 'id', got {positive!r}")
    return precision, recall, thresholds


def auroc(scores: Tensor, labels: Tensor) -> Tensor:
    """
    Area under the ROC curve (see :func:`roc_curve`), the probability that a random OOD sample
    has a higher score than a random ID sample, where ties count half.

    :param scores: outlier scores of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :return: AUROC as a float64 scalar tensor
    :raises ValueError: if the shapes differ, the scores contain NaN, or there are no ID or no
        OOD samples
    """
    fpr, tpr = _roc(_counts(scores, labels))
    return _area(fpr, tpr)


def aupr(scores: Tensor, labels: Tensor, positive: str = "ood") -> Tensor:
    """
    Area under the precision-recall curve (see :func:`pr_curve`), by the trapezoidal rule, as
    ``sklearn.metrics.auc(recall, precision)``.

    :param scores: outlier scores of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :param positive: the positive class, ``"ood"`` (AUPR-OUT) or ``"id"`` (AUPR-IN)
    :return: AUPR as a float64 scalar tensor
    :raises ValueError: if the shapes differ, the scores contain NaN, there are no ID or no OOD
        samples, or ``positive`` is invalid
    """
    return _aupr(_counts(scores, labels), positive)


def fpr_at_tpr(scores: Tensor, labels: Tensor, tpr: float = 0.95) -> Tensor:
    """
    False positive rate at the first threshold of the ROC curve (see :func:`roc_curve`) with a
    true positive rate of at least ``tpr``.

    :param scores: outlier scores of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :param tpr: true positive rate, a fraction in :math:`[0, 1]`
    :return: false positive rate as a float64 scalar tensor
    :raises ValueError: if the shapes differ, the scores contain NaN, there are no ID or no OOD
        samples, or ``tpr`` is not in :math:`[0, 1]`
    """
    _check_fraction("tpr", tpr)
    return _fpr_at_tpr(_counts(scores, labels), tpr)


def autc(scores: Tensor, labels: Tensor) -> Tensor:
    """
    Area under the threshold curve (AUTC): the mean of the areas under the false positive rate
    and the false negative rate curves, over thresholds :math:`\\tau \\in [0, 1]` on the min-max
    normalized scores. Lower is better. For the normalized scores :math:`s_i`,

    .. math::

        \\int_0^1 \\mathrm{FPR}(\\tau)\\,d\\tau = \\frac{1}{N_-}\\sum_{i:\\,y_i=0} s_i,
        \\qquad
        \\int_0^1 \\mathrm{FNR}(\\tau)\\,d\\tau = \\frac{1}{N_+}\\sum_{i:\\,y_i=1} (1 - s_i),

    where :math:`y_i = 1` marks the :math:`N_+` OOD and :math:`y_i = 0` the :math:`N_-` ID
    samples.

    :param scores: outlier scores of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :return: AUTC as a float64 scalar tensor
    :raises ValueError: if the shapes differ, the scores are not finite or constant, or there are
        no ID or no OOD samples

    :see Paper: `AUTC (arXiv 2306.14658) <https://arxiv.org/pdf/2306.14658>`__
    """
    scores, labels = _flatten(scores, labels)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("AUTC requires finite scores")
    ood = labels < 0
    n_ood = int(ood.sum())
    if n_ood == 0 or n_ood == ood.numel():
        raise ValueError("AUTC requires both ID and OOD samples")
    scores = _float64(scores)
    low, high = scores.min(), scores.max()
    if bool(low == high):
        raise ValueError("AUTC is undefined for constant scores")
    scores = (scores - low) / (high - low)
    ood = ood.to(scores.device)
    return (scores[~ood].mean() + (1 - scores[ood]).mean()) / 2


def accuracy(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Closed-set accuracy: the fraction of ID samples (labels :math:`\\geq 0`) whose predicted
    class equals the label. OOD samples are ignored.

    :param predictions: predicted class indices of any shape
    :param labels: labels of the same shape; labels :math:`< 0` mark OOD samples
    :return: accuracy as a float64 scalar tensor
    :raises ValueError: if the shapes differ or there are no ID samples
    """
    predictions, labels = _flatten(predictions, labels)
    known = labels >= 0
    n_known = int(known.sum())
    if n_known == 0:
        raise ValueError("Accuracy requires ID samples (labels >= 0)")
    correct = (predictions[known] == labels[known]).sum()
    return _float64(correct) / n_known


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


def calc_openness(n_train, n_test, n_target):
    """
    Openness of an open set recognition problem, as defined in *Toward Open Set Recognition*:

    .. math::
        1 - \\sqrt{ \\frac{2 \\, n_{\\text{train}}}{n_{\\text{test}} + n_{\\text{target}}} }

    It is :math:`0` for a closed set problem, where all classes seen during testing are known from
    training, and approaches :math:`1` as more unknown classes are added during testing.

    :param n_train: number of classes seen during training :math:`n_{\\text{train}}`
    :param n_test: total number of classes seen during testing :math:`n_{\\text{test}}`
    :param n_target: number of classes to recognize during testing :math:`n_{\\text{target}}`

    :return: openness of the problem

    :see Paper: `IEEE Explore <https://ieeexplore.ieee.org/abstract/document/6365193>`__
    """
    frac = 2 * n_train / (n_test + n_target)
    return 1 - math.sqrt(frac)
