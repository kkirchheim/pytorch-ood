"""

..  autoclass:: pytorch_ood.detector.NNGuide
    :members:

"""

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import Self

from pytorch_ood.api import (
    DetectorInfo,
    FeaturesDetector,
    ModelNotSetException,
    Paper,
    RequiresFittingException,
    Task,
)
from pytorch_ood.utils import extract_features, is_known
from pytorch_ood.utils.neighbors import (
    NearestNeighborSearch,
    TorchNearestNeighbors,
    _check_k,
    _check_search,
)

log = logging.getLogger(__name__)


class NNGuide(FeaturesDetector):
    """
    Implements *Nearest Neighbor Guidance for Out-of-Distribution Detection*.

    Guides the energy score with the similarity to a bank of training features. The bank holds
    the normalized features :math:`\\hat{f}(z) = f(z) / \\lVert f(z) \\rVert_2` of the training samples
    :math:`z`, each scaled by its energy :math:`E(z)`. The guidance is the mean of the :math:`k`
    largest inner products between the normalized features of the input and the bank, and the
    outlier score is the guidance times the energy of the input, negated relative to the paper:

    .. math::
        - \\underbrace{\\frac{1}{k} \\sum_{z \\in \\mathcal{N}_k(x)}
        \\langle \\hat{f}(x),\\, E(z) \\, \\hat{f}(z) \\rangle}_{\\text{guidance}} \\cdot E(x)

    where :math:`E(x) = \\log \\sum_i \\exp(l_i(x))` is the energy of the logits :math:`l(x)` and
    :math:`\\mathcal{N}_k(x)` are the :math:`k` training samples with the largest inner products
    :math:`\\langle \\hat{f}(x), E(z) \\, \\hat{f}(z) \\rangle`.

    The paper builds the bank from a random subset of the training data (e.g. 1% for ImageNet),
    which you can do by fitting on a subset.

    The :math:`k` largest inner products are found by ``search``, by default an exact search
    with :class:`TorchNearestNeighbors <pytorch_ood.utils.TorchNearestNeighbors>`. That search
    keeps the bank in memory (:math:`N \\times D` values) on the detector's device, and :meth:`to`
    moves it. For a bank too large for the GPU, pin it to the CPU with
    ``search=TorchNearestNeighbors(device="cpu", metric="inner_product")``; the search then runs
    on the CPU.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from pytorch_ood.detector import NNGuide
        from pytorch_ood.utils import TorchNearestNeighbors

        encoder, head = torch.nn.Linear(10, 16), torch.nn.Linear(16, 3)
        # the bank stays on the CPU, also if the detector is moved to a GPU
        search = TorchNearestNeighbors(device="cpu", metric="inner_product")
        detector = NNGuide(encoder, head, k=5, search=search)
        detector.fit_features(torch.randn(100, 16), torch.zeros(100, dtype=torch.long))
        scores = detector(torch.randn(8, 10))
    """

    info = DetectorInfo(
        paper=Paper(
            title="Nearest Neighbor Guidance for Out-of-Distribution Detection",
            venue="ICCV",
            year=2023,
            url="https://arxiv.org/abs/2309.14888",
            code="https://github.com/roomo7time/nnguide",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    requires_fit = True

    def __init__(
        self,
        encoder: Callable[[Tensor], Tensor],
        head: Callable[[Tensor], Tensor],
        k: int = 10,
        search: Optional[NearestNeighborSearch] = None,
    ):
        """
        :param encoder: neural network that extracts penultimate-layer features :math:`f`
        :param head: callable that maps features to logits :math:`l` (e.g., a linear layer)
        :param k: number :math:`k` of nearest neighbors for guidance. It can be changed after
            fitting.
        :param search: instance of a
            :class:`NearestNeighborSearch <pytorch_ood.utils.NearestNeighborSearch>` with
            ``metric == "inner_product"`` (see there for writing your own). Defaults to
            :class:`TorchNearestNeighbors(metric="inner_product")
            <pytorch_ood.utils.TorchNearestNeighbors>`. Fitting the detector fits this instance
            and replaces its reference set, so give each detector its own instance.
        :raises ValueError: if ``k`` is below 1 or ``search`` uses another metric
        :raises TypeError: if ``k`` is not an integer or ``search`` is not an instance of
            :class:`NearestNeighborSearch <pytorch_ood.utils.NearestNeighborSearch>`
        """
        super(NNGuide, self).__init__()
        _check_k(k)
        self.k = k
        if search is None:
            search = TorchNearestNeighbors(metric="inner_product")
        _check_search(search, "inner_product", "NNGuide")
        self.encoder = encoder
        self.head = head
        #: The nearest-neighbor search, fitted with the energy-scaled bank of the ID samples.
        self.search = search
        self._is_fitted = False

    def fit(self, data_loader: DataLoader, device=None) -> Self:
        """
        Extract features from the data loader and build the energy-scaled feature bank.

        :param data_loader: data loader with ID training data
        :param device: device for feature extraction. If ``None``, the detector device is used, else the
            device of the encoder parameters, else ``cpu``.
        :return: self
        :raise ValueError: if the data contains fewer than :math:`k` ID samples
        """
        if device is None:
            device = self.device
            if device is None:
                if isinstance(self.encoder, torch.nn.Module):
                    device = next(self.encoder.parameters()).device
                else:
                    device = "cpu"
                log.warning(f"No device given. Will use '{device}'.")

        if isinstance(self.encoder, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            self.encoder.to(device)

        z, y = extract_features(model=self.encoder, data_loader=data_loader, device=device)
        return self.fit_features(z, y)

    def fit_features(self, z: Tensor, labels: Tensor, batch_size: int = 4096) -> Self:
        """
        Build the energy-scaled feature bank from pre-extracted features and fit the search
        with it.

        :param z: features of shape :math:`N \\times D`
        :param labels: class labels of shape :math:`N`; OOD samples (label below zero) are ignored
        :param batch_size: number of samples passed through ``head`` at once; lower it to reduce
            peak GPU memory
        :return: self
        :raise ValueError: if fewer than :math:`k` ID samples are given or :math:`k` is below 1
        :raise TypeError: if :math:`k` is not an integer
        """
        known = is_known(labels).to(z.device)
        _check_k(self.k, int(known.sum()), "ID samples")

        device = self.device or z.device
        z = z[known].detach().float()

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        # chunked, so that peak GPU memory scales with batch_size, not with the size of the bank
        chunks = []
        with torch.no_grad():
            for start in range(0, z.shape[0], batch_size):
                z_batch = z[start : start + batch_size].to(device)
                energy = torch.logsumexp(self.head(z_batch), dim=1)
                chunks.append(F.normalize(z_batch, dim=1) * energy[:, None])
        self.search.fit(torch.cat(chunks))
        self._is_fitted = True

        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        :param x: model inputs
        :return: outlier scores of shape :math:`B`
        :raise ModelNotSetException: if no encoder was given
        :raise RequiresFittingException: if the detector has not been fitted
        """
        if self.encoder is None:
            raise ModelNotSetException()
        if not self._is_fitted:
            raise RequiresFittingException()

        with torch.no_grad():
            z = self.encoder(x)
        return self.predict_features(z)

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute the NNGuide outlier score from pre-extracted features.

        :param z: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector has not been fitted
        :raise ValueError: if ``k`` is below 1 or exceeds the number of fitted samples
        :raise TypeError: if ``k`` is not an integer
        """
        if not self._is_fitted:
            raise RequiresFittingException()
        _check_k(self.k, self.search.n_samples, "fitted samples")

        device = self.device or z.device
        z = z.to(device)

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        energy = torch.logsumexp(self.head(z), dim=1)
        # the k largest inner products with the bank
        similarities, _ = self.search.search(F.normalize(z, dim=1), self.k)
        guidance = similarities.mean(dim=1)

        # higher guidance * energy = more in-distribution, so negate for outlier score
        return -(guidance * energy)

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move the detector's state to ``device``, including the bank in ``search`` unless the
        search is pinned to a device.

        :param device: target device, e.g. ``"cuda:0"`` or a :class:`torch.device`
        :return: self
        """
        super().to(device)
        # in place; the return value is not needed
        self.search.to(device)
        return self
