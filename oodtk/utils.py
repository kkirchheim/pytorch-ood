"""

"""
import logging
from collections import defaultdict

import torch
import torchmetrics

log = logging.getLogger(__name__)


def calc_openness(n_train, n_test, n_target):
    """
    In *Toward open set recognition* from Scheirer, Jain, Boult et al, the Openness was defined.

    .. math::
        \\mathcal{O} = 1 - \\sqrt{ \\frac{2 \\times  n_{train}}{n_{test} \\times n_{target}} }


    :return: Openness of the problem

    :see Paper: https://ieeexplore.ieee.org/abstract/document/6365193
    """
    frac = 2 * n_train / (n_test + n_target)
    return 1 - torch.sqrt(frac)


#######################################
# Helpers for labels
#######################################
def is_known(labels) -> bool:
    """
    :returns: True, if label :math:`>= 0`
    """
    return labels >= 0


def is_unknown(labels) -> bool:
    """
    :returns: True, if label :math:`< 0`
    """
    return labels < 0


def contains_known_and_unknown(labels) -> bool:
    """
    :return: true if the labels contain *IN* and *OOD* classes
    """
    return contains_known(labels) and contains_unknown(labels)


def contains_known(labels) -> bool:
    """
    :return: true if the labels contains any *IN* labels
    """
    return is_known(labels).any()


def contains_unknown(labels) -> bool:
    """
    :return: true if the labels contains any *OOD* labels
    """
    return is_unknown(labels).any()


#######################################
# Distance functions etc.
#######################################
def estimate_class_centers(
    embedding: torch.Tensor, target: torch.Tensor, num_centers: int = None
) -> torch.Tensor:
    """
    Estimates class centers from the given embeddings and labels, using mean as estimator.

    TODO: the loop can prob. be replaced
    """
    batch_classes = torch.unique(target).long().to(embedding.device)
    if num_centers is None:
        num_centers = torch.max(target) + 1
    centers = torch.zeros((num_centers, embedding.shape[1]), device=embedding.device)
    for clazz in batch_classes:
        centers[clazz] = embedding[target == clazz].mean(dim=0)
    return centers


def torch_get_distances(centers, embeddings):
    """
    TODO: this can be done more efficiently
    """
    n_instances = embeddings.shape[0]
    n_centers = centers.shape[0]
    distances = torch.empty((n_instances, n_centers)).to(embeddings.device)
    for clazz in torch.arange(n_centers):
        distances[:, clazz] = torch.norm(embeddings - centers[clazz], dim=1, p=2)
    return distances


def pairwise_distances(x, y=None) -> torch.Tensor:
    """
    Calculate pairwise distance by quadratic expansion.

    :param x: is a Nxd matrix
    :param y:  Mxd matrix

    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

    :see Implementation: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, torch.inf)


class TensorBuffer(object):
    """
    Used to buffer tensors
    """

    def __init__(self, device="cpu"):
        """

        :param device: device used to store buffers. Default is *cpu*.
        """
        self._buffer = defaultdict(list)
        self.device = device

    def append(self, key, value: torch.Tensor):
        """
        Appends a tensor to the buffer.

        :param key: tensor identifier
        :param value: tensor
        :return: self
        """
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Can not handle value type {type(value)}")

        value = value.detach().to(self.device)
        self._buffer[key].append(value)
        return self

    def __contains__(self, elem):
        return elem in self._buffer

    def __getitem__(self, item):
        return self.get(item)

    def sample(self, key) -> torch.Tensor:
        """
        Samples a random tensor from the buffer

        :param key: tensor identifier
        :return: random tensor
        """
        index = torch.randint(0, len(self._buffer[key]), size=(1,))
        return self._buffer[key][index]

    def get(self, key) -> torch.Tensor:
        """
        Retrieves tensor from the buffer

        :param key: tensor identifier
        :return: concatenated tensor
        """
        if key not in self._buffer:
            raise KeyError(key)

        v = torch.cat(self._buffer[key])
        return v

    def clear(self):
        """
        Clears the buffer

        :return: self
        """
        log.debug("Clearing buffer")
        self._buffer.clear()
        return self

    def save(self, path):
        """
        Save buffer to disk

        :return: self
        """
        d = {k: self.get(k).cpu() for k in self._buffer.keys()}
        torch.save(d, path)
        return self


####################################
# METRICS
####################################


def fpr_at_tpr(pred, target, k=0.95):
    """

    TODO: use bisect

    :param pred:
    :param target:
    :param k:
    :return:
    """
    fpr, tpr, thresholds = torchmetrics.functional.roc(pred, target)
    for fp, tp, t in zip(fpr, tpr, thresholds):
        if tp >= k:
            return fp


def accuracy_at_tpr(pred, target, k=0.95):
    """
    TODO: use bisect

    :param pred:
    :param target:
    :param k:
    :return:
    """
    fpr, tpr, thresholds = torchmetrics.functional.roc(pred, target)
    for fp, tp, t in zip(fpr, tpr, thresholds):
        if tp >= k:
            break

    labels = torch.where(pred >= t, 1, 0)
    return torchmetrics.functional.accuracy(labels, target)


class OODMetrics(torch.nn.Module):
    """
    Calculates various metrics used in OOD.

    .. note :: This implementation is not optimized.
    """

    def __init__(self):
        super(OODMetrics, self).__init__()
        self.buffer = TensorBuffer()
        self.auroc = torchmetrics.AUROC(num_classes=2)
        self.aupr_in = torchmetrics.PrecisionRecallCurve(pos_label=1)
        self.aupr_out = torchmetrics.PrecisionRecallCurve(pos_label=0)

    def update(self, outlier_scores, y):
        label = is_unknown(y)
        self.auroc.update(outlier_scores, label)
        self.aupr_in.update(outlier_scores, label)
        self.aupr_out.update(-outlier_scores, label)
        self.buffer.append("scores", outlier_scores)
        self.buffer.append("labels", label)

    def compute(self) -> dict:
        auroc = self.auroc.compute()

        p, r, t = self.aupr_in.compute()
        aupr_in = torchmetrics.functional.auc(r, p, reorder=True)

        p, r, t = self.aupr_out.compute()
        aupr_out = torchmetrics.functional.auc(r, p)

        label = self.buffer.get("labels")
        s = self.buffer.get("scores")

        acc = accuracy_at_tpr(s, label)
        fpr = fpr_at_tpr(s, label)

        return {
            "AUROC": auroc.item(),
            "AUPR-IN": aupr_in.item(),
            "AUPR-OUT": aupr_out.item(),
            "ACC95TPR": acc.item(),
            "FPR95TPR": fpr.item(),
        }

    def reset(self):
        self.auroc.reset()
        self.aupr_in.reset()
        self.aupr_out.reset()
        self.buffer.clear()
