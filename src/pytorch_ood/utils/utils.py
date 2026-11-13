""" """

import logging
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    KeysView,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from numpy import floating
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from typing_extensions import Self

from ..api import ModelModeException

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


@contextmanager
def _requires_grad(params: List[Tensor]) -> Iterator[None]:
    """Temporarily enables gradients for ``params``, so that frozen parameters can be scored."""
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def _require_eval_mode(model: Module, detector: object) -> None:
    """Raises :class:`ModelModeException` if ``model`` or one of its submodules is in training mode."""
    training = [(name, module) for name, module in model.named_modules() if module.training]
    if not training:
        return
    # name the layers rather than the containers that hold them
    names = [name for name, module in training if not any(module.children())]
    names = names or [name or "the model itself" for name, _ in training]
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    raise ModelModeException(
        f"{type(detector).__name__} needs every module of the model in evaluation mode, call "
        f"model.eval() first. In training mode: {shown}"
    )


def _check_fraction(name: str, value: float) -> float:
    """
    Percentiles and fractions are given in :math:`[0, 1]` throughout the library;
    raise for values outside, with a hint for values given in percent.
    """
    if not 0.0 <= value <= 1.0:
        hint = f"; use {value / 100:g} instead of {value:g}" if 1.0 < value <= 100.0 else ""
        raise ValueError(f"{name} must be a fraction in [0, 1], got {value}{hint}")
    return value


def _linalg_dtype(device) -> torch.dtype:
    """
    Dtype for covariance estimates, inverses and log-determinants: float64, since float32 breaks down
    on ill-conditioned matrices; MPS does not support float64.
    """
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def _gram_matrix(x: Tensor, transform: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
    """
    :math:`X^\\top X` in :func:`_linalg_dtype`, computed in chunks of rows so that only one chunk at a
    time is held in that dtype.

    :param x: matrix :math:`X` of shape :math:`N \\times D`
    :param transform: applied to each chunk of rows after the conversion, e.g. centering
    :return: matrix of shape :math:`D \\times D`
    """
    dtype = _linalg_dtype(x.device)
    gram = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=dtype)
    # 16384 rows of 2048 float64 features take 256 MB
    for chunk in x.detach().split(16384):
        chunk = chunk.to(dtype)
        if transform is not None:
            chunk = transform(chunk)
        gram += chunk.T @ chunk
    return gram


@dataclass
class _Gaussian:
    """
    Gaussians with class means and a shared covariance, as fitted by :func:`_fit_gaussian`.
    """

    mu: Tensor  #: class means, :math:`C \\times D`
    cov: Tensor  #: shared covariance (regularized unless ``use_pinv``), :math:`D \\times D`
    precision: Tensor  #: (pseudo-)inverse of ``cov``, :math:`D \\times D`
    #: :math:`L` of shape :math:`D \\times R` with ``precision`` :math:`= L L^\\top`
    whitening: Tensor
    whitened_mu: Tensor  #: ``mu @ whitening``, :math:`C \\times R`


def _fit_gaussian(z: Tensor, y: Tensor, n_classes: int, eps: float, use_pinv: bool) -> _Gaussian:
    """
    Class means and shared within-class covariance (centered outer products divided by :math:`N`,
    as sklearn's ``EmpiricalCovariance``), plus its inverse and a factor :math:`L` of the inverse.
    Computed in :func:`_linalg_dtype`, returned in the dtype of ``z``.

    With ``use_pinv``, the pseudo-inverse discards directions without variance and ``eps`` is not
    used: a ridge would turn them into directions of precision ``1 / eps``, and whether ``pinv`` keeps
    those depends on its relative cutoff, i.e. on the scale of the features.

    :param z: features of shape :math:`N \\times D`
    :param y: class labels of shape :math:`N`, in :math:`0, \\ldots, C-1`
    :param n_classes: number of classes :math:`C`; all of them have to occur
    :param eps: added to the diagonal of the covariance before inversion
    :param use_pinv: use the pseudo-inverse instead of the inverse of the regularized covariance
    :return: the fitted Gaussians
    :raises ValueError: if not ``use_pinv`` and the regularized covariance is not positive definite
    """
    dtype = _linalg_dtype(z.device)
    z = z.detach()
    y = y.long()
    d = z.shape[1]
    mu = torch.zeros(n_classes, d, device=z.device, dtype=dtype)
    cov = torch.zeros(d, d, device=z.device, dtype=dtype)
    # per class, so only one class at a time is held in float64
    for k in range(n_classes):
        z_k = z[y == k].to(dtype)
        mu[k] = z_k.mean(dim=0)
        centered = z_k - mu[k]
        cov += centered.T @ centered
    cov /= z.shape[0]
    if not use_pinv:
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=dtype)
    # Factor the covariance, not the precision: cholesky(inv(cov)) fails when cov is rank-deficient
    # (fewer samples than dimensions, dead units), since inv(cov + eps I) then has eigenvalues ~1/eps
    # and is not numerically positive definite.
    if cov.device.type == "mps":
        # eigh has no MPS kernel
        lam, vec = (t.to(cov.device) for t in torch.linalg.eigh(cov.cpu()))
    else:
        lam, vec = torch.linalg.eigh(cov)
    if use_pinv:
        # the default cutoff of torch.linalg.pinv(hermitian=True)
        keep = lam > torch.finfo(dtype).eps * d * lam.abs().max()
        lam, vec = lam[keep], vec[:, keep]
    elif lam.min() <= 0:
        raise ValueError(
            "The covariance matrix plus eps * I is not positive definite (smallest eigenvalue "
            f"{lam.min():.1e}, eps={eps}); increase eps or use use_pinv=True"
        )
    whitening = vec * lam.rsqrt()
    return _Gaussian(
        mu=mu.to(z.dtype),
        cov=cov.to(z.dtype),
        precision=(whitening @ whitening.T).to(z.dtype),
        whitening=whitening.to(z.dtype),
        whitened_mu=(mu @ whitening).to(z.dtype),
    )


def _squared_mahalanobis(z: Tensor, whitening: Tensor, whitened_mu: Tensor) -> Tensor:
    """
    Squared Mahalanobis distances of every sample to every mean, as Euclidean distances in the
    whitened space: :math:`\\|(z - \\mu_k) L\\|^2` with precision :math:`L L^\\top`.

    :param z: features of shape :math:`B \\times D`
    :param whitening: :attr:`_Gaussian.whitening` :math:`L`, shape :math:`D \\times R`
    :param whitened_mu: :attr:`_Gaussian.whitened_mu`, shape :math:`C \\times R`
    :return: distances of shape :math:`B \\times C`
    """
    # Costs B*D*R + B*C*R instead of C*B*D^2 for a loop over classes. The direct mode avoids
    # cdist's |a|^2 - 2ab + |b|^2 expansion, which cancels catastrophically near a class mean.
    # cdist runs in _linalg_dtype: on the CPU it sums the R squared differences sequentially, which
    # in float32 costs ~1e-6 relative error at R=2048, amplified ~100x by RMD's difference of
    # distances. The B*C*R work is small next to the projection.
    w = z @ whitening
    dtype = _linalg_dtype(w.device)
    d = torch.cdist(
        w.to(dtype), whitened_mu.to(dtype), compute_mode="donot_use_mm_for_euclid_dist"
    )
    return d.square().to(z.dtype)


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


def drop_unknown(target: Tensor, *tensors: Tensor) -> Tuple[Tensor, ...]:
    """
    Discards the OOD samples (targets :math:`< 0`) of a batch, with a warning. Losses that do not use
    OOD samples call this at the start of ``forward``, so that they compute the same loss as on a
    batch without these samples.

    :param target: labels of shape :math:`B`
    :param tensors: tensors of the batch, each with first dimension :math:`B`
    :return: tuple ``(target, *tensors)``, restricted to the samples with targets :math:`\\geq 0`
    :raises ValueError: if ``target`` is not one-dimensional
    """
    if target.dim() != 1:
        raise ValueError(f"Expected targets of shape B, got {tuple(target.shape)}")

    known = is_known(target)
    if known.all():
        return (target, *tensors)

    warnings.warn(
        "Discarding samples with targets < 0 (OOD), which this loss does not use.", stacklevel=2
    )
    return (target[known], *(t[known] for t in tensors))


_NO_ID_SAMPLES = "No ID samples to fit on (all labels are < 0)"


def _known_for_fit(labels: Tensor) -> Tensor:
    """
    :func:`is_known` for the labels of fitting data, with a warning if there are OOD samples.
    The message is constant, so Python shows it once per call site.
    """
    known = is_known(labels)
    if not known.all():
        warnings.warn(
            "Ignoring samples with labels < 0 (OOD) for fitting; this detector uses only ID samples.",
            stacklevel=3,
        )
    return known


def _id_rows_for_fit(representation, labels: Tensor):
    """
    The ID samples of fitting data, see :func:`_known_for_fit`.

    :param representation: tensor, or list of tensors, with first dimension :math:`N`
    :param labels: labels of shape :math:`N`
    :return: tuple ``(representation, labels)`` restricted to the ID samples
    :raises ValueError: if there is no ID sample
    """
    known = _known_for_fit(labels)
    if known.all():
        return representation, labels
    if not known.any():
        raise ValueError(_NO_ID_SAMPLES)

    def rows(t: Tensor) -> Tensor:
        return t[known.to(t.device)]

    if isinstance(representation, (list, tuple)):
        return [rows(t) for t in representation], labels[known]
    return rows(representation), labels[known]


def _id_batches(batches: Iterable) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
    """
    The ID samples of each batch ``(x, y)`` of fitting data, see :func:`_known_for_fit`. Batches
    without ID samples are skipped; ``y`` may be ``None``, then the batch is used as it is.

    :raises ValueError: if there is no ID sample
    """
    found = False
    for x, y in batches:
        if y is not None:
            known = _known_for_fit(y)
            if not known.any():
                continue
            if not known.all():
                x, y = x[known], y[known]
        found = True
        yield x, y
    if not found:
        raise ValueError(_NO_ID_SAMPLES)


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
        for x, y in _id_batches(data_loader):
            z = model(x.to(device))
            # average and flatten
            buffer.append("embedding", z.mean(dim=(2, 3)).view(len(z), -1))
            buffer.append("label", y.to(device))

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
        for x, y in _id_batches(data_loader):
            z = model(x.to(device))
            buffer.append("embedding", z.view(len(z), -1))  # flatten
            buffer.append("label", y.to(device))

    z = buffer.get("embedding")
    y = buffer.get("label")
    return z, y


@torch.no_grad()
def apply_in_batches(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    batch_size: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """
    Apply ``fn`` to blocks of ``batch_size`` rows of ``x`` on ``device``, so that memory there
    scales with ``batch_size``, and collect the outputs in one tensor on the device of ``x``,
    allocated once.

    :param fn: function that maps a block of :math:`B` rows to :math:`B` outputs
    :param x: input of shape :math:`N \\times \\dots`
    :param batch_size: number of rows per block
    :param device: device to apply ``fn`` on. If ``None``, the blocks are not moved.
    :return: outputs of shape :math:`N \\times \\dots` on the device of ``x``
    :raises ValueError: if ``x`` has no rows, or ``fn`` does not return one output per row
    """
    if x.shape[0] == 0:
        raise ValueError("x must have at least one row")
    out = None
    for start in range(0, x.shape[0], batch_size):
        block = x[start : start + batch_size].to(device)
        y = fn(block)
        if y.shape[0] != block.shape[0]:
            raise ValueError(
                f"fn must return one output per row, got {y.shape[0]} for {block.shape[0]} rows"
            )
        if out is None:
            out = y.new_empty((x.shape[0], *y.shape[1:]), device=x.device)
        out[start : start + y.shape[0]] = y
    return out


def to_np(x: Tensor):
    return x.data.cpu().numpy()


def evaluate_energy_logistic_loss(
    model: Callable[[Tensor], Tensor],
    train_loader_in: DataLoader,
    logistic_regression: Callable[[Tensor], Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[floating, floating, floating]:
    """
    Evaluate energy logistic loss on ID training dataset

    :param model: neural network to pass inputs to
    :param train_loader_in: dataset to extract from
    :param logistic_regression: logistic regression layer
    :param device: device of ``model``, to which the batches are moved; if not given, the batches stay
        where the loader puts them
    :return: average sigmoid energy loss, average logistic energy loss and average cross-entropy
    """
    model.eval()
    sigmoid_energy_losses = []
    logistic_energy_losses = []
    ce_losses = []
    with torch.no_grad():
        for in_set in train_loader_in:
            data = in_set[0]
            target = in_set[1]

            if device is not None:
                data, target = data.to(device), target.to(device)

            # forward
            y = model(data)

            # compute energies
            Ec_in = torch.logsumexp(y, dim=1)

            # compute labels
            binary_labels_1 = torch.ones_like(Ec_in)

            # compute in distribution logistic losses
            # squeeze(1), so that a batch of size one keeps its batch dimension
            logistic_loss_energy_in = F.binary_cross_entropy_with_logits(
                logistic_regression(Ec_in.unsqueeze(1)).squeeze(1),
                binary_labels_1,
                reduction="none",
            )

            logistic_energy_losses.extend(list(to_np(logistic_loss_energy_in)))

            # compute in distribution sigmoid losses
            sigmoid_loss_energy_in = torch.sigmoid(
                logistic_regression(Ec_in.unsqueeze(1)).squeeze(1)
            )

            sigmoid_energy_losses.extend(list(to_np(sigmoid_loss_energy_in)))

            # in-distribution classification losses
            loss_ce = F.cross_entropy(y, target, reduction="none")

            ce_losses.extend(list(to_np(loss_ce)))

    avg_sigmoid_energy_losses = np.mean(np.array(sigmoid_energy_losses))

    avg_logistic_energy_losses = np.mean(np.array(logistic_energy_losses))

    avg_ce_loss = np.mean(np.array(ce_losses))

    return avg_sigmoid_energy_losses, avg_logistic_energy_losses, avg_ce_loss
