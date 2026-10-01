""" """

import logging
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, KeysView, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy import floating
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

log = logging.getLogger(__name__)


def temperature_calibration(
    logits: Tensor,
    labels: Tensor,
    lower: float = 0.2,
    upper: float = 5.0,
    eps: float = 0.0001,
) -> float:
    """
    Implements confidence calibration from the paper
    *On Calibration of Modern Neural Networks*.

    Finds the temperature :math:`T` that minimizes the cross-entropy of the scaled logits
    :math:`z / T`, using binary search.

    :see Paper: `On Calibration of Modern Neural Networks (ICML 2017) <http://proceedings.mlr.press/v70/guo17a.html>`__
    :see Implementation: `PixMix calibration_tools.py <https://github.com/andyzoujm/pixmix/blob/main/calibration_tools.py>`__

    :param logits: the logits predicted by the model, of shape :math:`N \\times C`
    :param labels: ground truth class labels of shape :math:`N`
    :param lower: lower bound for the search
    :param upper: upper bound for the search
    :param eps: minimum change necessary to continue optimization
    :return: temperature :math:`T` in ``[lower, upper]`` that minimizes the cross-entropy of
        ``logits / T``
    """
    logits = torch.FloatTensor(logits)
    labels = torch.LongTensor(labels)
    t_guess = torch.FloatTensor([0.5 * (lower + upper)]).requires_grad_()

    while upper - lower > eps:
        if torch.autograd.grad(F.cross_entropy(logits / t_guess, labels), t_guess)[0] > 0:
            upper = 0.5 * (lower + upper)
        else:
            lower = 0.5 * (lower + upper)
        t_guess = t_guess * 0 + 0.5 * (lower + upper)

    t = min(
        [lower, 0.5 * (lower + upper), upper],
        key=lambda x: float(F.cross_entropy(logits / x, labels)),
    )

    return t


def calc_openness(n_train, n_test, n_target):
    """
    In *Toward open set recognition* the Openness  :math:`\\mathcal{O}`  of a problem was defined as:

    .. math::
        \\mathcal{O} = 1 - \\sqrt{ \\frac{2 \\times  n_{train}}{n_{test} \\times n_{target}} }

    where :math:`n` is the number of classes, respectively.

    :param n_train: number of classes for training
    :param n_test: total number of classes used during testing
    :param n_target: number of classes for classification during testing

    :return: Openness of the problem

    :see Paper: `IEEE Explore <https://ieeexplore.ieee.org/abstract/document/6365193>`__
    """
    frac = 2 * n_train / (n_test + n_target)
    return 1 - math.sqrt(frac)


def _check_fraction(name: str, value: float) -> float:
    """
    Percentiles and fractions are given in :math:`[0, 1]` throughout the library;
    raise for values outside, with a hint for values given in percent.
    """
    if not 0.0 <= value <= 1.0:
        hint = f"; use {value / 100:g} instead of {value:g}" if 1.0 < value <= 100.0 else ""
        raise ValueError(f"{name} must be a fraction in [0, 1], got {value}{hint}")
    return value


#######################################
# Helpers for labels
#######################################
def is_known(labels) -> Union[bool, Tensor]:
    """
    Checks which labels belong to *ID* samples.

    :param labels: tensor of labels
    :return: boolean mask of the shape of ``labels``, True where the label is :math:`\\geq 0`
    """
    return labels >= 0


def is_unknown(labels) -> Union[bool, Tensor]:
    """
    Checks which labels belong to *OOD* samples.

    :param labels: tensor of labels
    :return: boolean mask of the shape of ``labels``, True where the label is :math:`< 0`
    """
    return labels < 0


def contains_known_and_unknown(labels) -> Union[bool, Tensor]:
    """
    :param labels: tensor of labels
    :return: True if the labels contain *ID* and *OOD* classes (as a 0-d boolean tensor)
    """
    return contains_known(labels) and contains_unknown(labels)


def contains_known(labels) -> Union[bool, Tensor]:
    """
    :param labels: tensor of labels
    :return: True if the labels contain any *ID* labels (as a 0-d boolean tensor)
    """
    return is_known(labels).any()


def contains_unknown(labels) -> Union[bool, Tensor]:
    """
    :param labels: tensor of labels
    :return: True if the labels contain any *OOD* labels (as a 0-d boolean tensor)
    """
    return is_unknown(labels).any()


#######################################
# Distance functions etc.
#######################################
def estimate_class_centers(embedding: Tensor, target: Tensor, num_centers: int = None) -> Tensor:
    """
    Estimates class centers from the given embeddings and labels, using mean as estimator.

    Labels must be :math:`\\geq 0`; OOD samples have to be removed beforehand.

    :param embedding: embeddings of shape :math:`N \\times D`
    :param target: class labels of shape :math:`N`
    :param num_centers: number of classes :math:`K`; defaults to ``max(target) + 1``
    :return: class centers of shape :math:`K \\times D`. Rows of classes that do not occur in
        ``target`` are zero.
    """
    # TODO: the loop can prob. be replaced
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


def pairwise_distances(x: Tensor, y: Tensor = None) -> Tensor:
    """
    Calculate pairwise squared Euclidean distance by quadratic expansion.

    :param x: is a :math:`N \\times D` matrix
    :param y:  :math:`M \\times D` matrix
    :returns: a :math:`N \\times M` matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

    :see Implementation: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


class TensorBuffer(object):
    """
    Collects tensors (e.g., batches) under keys and returns them concatenated along the first
    dimension. Tensors are detached and stored on ``device``.
    """

    def __init__(self, device="cpu"):
        """
        :param device: device used to store buffers. Default is *cpu*.
        """
        self._buffer: Dict[Any, Tensor] = defaultdict(list)
        self.device = device

    def is_empty(self) -> bool:
        """
        Returns true if this buffer does not hold any tensors.
        """
        return len(self._buffer) == 0

    def append(self, key, value: Tensor) -> Self:
        """
        Appends a tensor to the buffer.

        :param key: tensor identifier
        :param value: tensor
        :return: self
        :raises ValueError: if ``value`` is not a tensor
        """
        if not isinstance(value, Tensor):
            raise ValueError(f"Can not handle value type {type(value)}")

        value = value.detach().to(self.device)
        self._buffer[key].append(value)
        return self

    def __contains__(self, elem) -> bool:
        return elem in self._buffer

    def __getitem__(self, item) -> Tensor:
        return self.get(item)

    def sample(self, key) -> Tensor:
        """
        Samples a random tensor from the buffer, i.e., one of the tensors (batches) that were
        appended under ``key``, not a single row.

        :param key: tensor identifier
        :return: random tensor
        """
        index = torch.randint(0, len(self._buffer[key]), size=(1,))
        return self._buffer[key][index]

    def keys(self) -> KeysView:
        """
        :return: the identifiers of the stored tensors
        """
        return self._buffer.keys()

    def get(self, key) -> Tensor:
        """
        Retrieves tensor from the buffer

        :param key: tensor identifier
        :return: concatenated tensor
        :raises KeyError: if no tensor was appended under ``key``
        """
        if key not in self._buffer:
            raise KeyError(key)

        v = torch.cat(self._buffer[key])
        return v

    def clear(self) -> Self:
        """
        Clears the buffer
        """
        log.debug("Clearing buffer")
        self._buffer.clear()
        return self

    def save(self, path) -> Self:
        """
        Save buffer to disk, as a dictionary mapping the identifiers to the concatenated tensors

        :param path: file to write to
        :return: self
        """
        d = {k: self.get(k).cpu() for k in self._buffer.keys()}
        torch.save(d, path)
        return self


def apply_reduction(tensor: Tensor, reduction: str) -> Tensor:
    """
    Apply specific reduction to a tensor
    """
    if reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    elif reduction is None or reduction == "none":
        return tensor
    else:
        raise ValueError


def fix_random_seed(seed: int = 12345) -> None:
    """
    Seeds torch (CPU and all CUDA devices), :mod:`random` and numpy. This does not make cuDNN
    kernels deterministic.

    :param seed: seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def extract_feature_avg(
    data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: Optional[str]
) -> Tuple[Tensor, Tensor]:
    """
    Helper to extract features from model. Will compute mean over feature maps. Ignores OOD inputs.
    The model is not switched to evaluation mode.

    :param data_loader: loader yielding batches ``(x, y)`` to extract from
    :param model: callable mapping inputs to feature maps of shape
        :math:`B \\times C \\times H \\times W`
    :param device: device used for calculations. If ``None``, inputs are not moved.
    :return: tuple ``(z, y)`` with the :math:`N \\times C` spatially averaged features and the
        :math:`N` labels of the ID samples
    :raises ValueError: if the loader contains no ID samples (labels :math:`\\geq 0`)
    """
    # TODO: add option to buffer to GPU
    buffer = TensorBuffer()

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            known = is_known(y)
            if known.any():
                z = model(x[known])

                # average and flatten
                z = z.mean(dim=(2, 3)).view(known.sum(), -1)
                buffer.append("embedding", z)
                buffer.append("label", y[known])

        if buffer.is_empty():
            raise ValueError("No ID instances in loader")

    z = buffer.get("embedding")
    y = buffer.get("label")
    return z, y


def extract_features(
    data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: Optional[str]
) -> Tuple[Tensor, Tensor]:
    """
    Helper to extract outputs from model. Ignores OOD inputs.
    The model is not switched to evaluation mode.

    :param data_loader: loader yielding batches ``(x, y)`` to extract from
    :param model: callable mapping inputs to outputs, which are flattened per sample
    :param device: device used for calculations. If ``None``, inputs are not moved.
    :return: tuple ``(z, y)`` with the :math:`N \\times D` outputs and the :math:`N` labels of the
        ID samples
    :raises ValueError: if the loader contains no ID samples (labels :math:`\\geq 0`)
    """
    # TODO: add option to buffer to GPU
    buffer = TensorBuffer()

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            known = is_known(y)
            if known.any():
                z = model(x[known])
                z = z.view(known.sum(), -1)  # flatten
                buffer.append("embedding", z)
                buffer.append("label", y[known])

        if buffer.is_empty():
            raise ValueError("No ID instances in loader")

    z = buffer.get("embedding")
    y = buffer.get("label")
    return z, y


def to_np(x: Tensor):
    return x.data.cpu().numpy()


def evaluate_energy_logistic_loss(
    model: Callable[[Tensor], Tensor],
    train_loader_in: DataLoader,
    logistic_regression: Callable[[Tensor], Tensor],
) -> Tuple[floating, floating, floating]:
    """
    Evaluate energy logistic loss on ID training dataset

    :param model: neural network to pass inputs to
    :param train_loader_in: dataset to extract from
    :param logistic_regression: logistic regression layer
    :return: ndarray with average loss
    """
    model.eval()
    sigmoid_energy_losses = []
    logistic_energy_losses = []
    ce_losses = []
    for in_set in train_loader_in:
        data = in_set[0]
        target = in_set[1]

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # forward
        y = model(data)

        # compute energies
        Ec_in = torch.logsumexp(y, dim=1)

        # compute labels
        binary_labels_1 = torch.ones(len(data)).cuda()

        # compute in distribution logistic losses
        logistic_loss_energy_in = F.binary_cross_entropy_with_logits(
            logistic_regression(Ec_in.unsqueeze(1)).squeeze(),
            binary_labels_1,
            reduction="none",
        )

        logistic_energy_losses.extend(list(to_np(logistic_loss_energy_in)))

        # compute in distribution sigmoid losses
        sigmoid_loss_energy_in = torch.sigmoid(logistic_regression(Ec_in.unsqueeze(1)).squeeze())

        sigmoid_energy_losses.extend(list(to_np(sigmoid_loss_energy_in)))

        # in-distribution classification losses
        loss_ce = F.cross_entropy(y, target, reduction="none")

        ce_losses.extend(list(to_np(loss_ce)))

    avg_sigmoid_energy_losses = np.mean(np.array(sigmoid_energy_losses))

    avg_logistic_energy_losses = np.mean(np.array(logistic_energy_losses))

    avg_ce_loss = np.mean(np.array(ce_losses))

    return avg_sigmoid_energy_losses, avg_logistic_energy_losses, avg_ce_loss
