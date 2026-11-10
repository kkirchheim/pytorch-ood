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

from ..api import (
    DetectorInfo,
    FeaturesDetector,
    Paper,
    RequiresFittingException,
    Task,
)
from ..utils import apply_in_batches, extract_features, is_known
from ..utils.neighbors import NearestNeighborSearch, TorchNearestNeighbors, _check_k, _check_search

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

    The neighbors are found by ``search``, which keeps the bank on the detector's device. For a
    bank too large for the GPU, pin the search to the CPU, as in the example.

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
        :param k: number :math:`k` of nearest neighbors for guidance; can be changed after fitting
        :param search: a :class:`NearestNeighborSearch <pytorch_ood.utils.NearestNeighborSearch>`
            with ``metric == "inner_product"``, refitted by each fit (use one per detector).
            Defaults to :class:`TorchNearestNeighbors <pytorch_ood.utils.TorchNearestNeighbors>`.
        :raises ValueError: if ``k`` is below 1 or ``search`` uses another metric
        :raises TypeError: if ``k`` is not an integer or ``search`` is not a search
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
        :raises ModelNotSetException: if no encoder was given
        :raises ValueError: if the data contains fewer than :math:`k` ID samples
        """
        encoder = self._require_producer()
        if device is None:
            device = self.device
            if device is None:
                if isinstance(encoder, torch.nn.Module):
                    device = next(encoder.parameters()).device
                else:
                    device = "cpu"
                log.warning(f"No device given. Will use '{device}'.")

        if isinstance(encoder, torch.nn.Module):
            log.debug(f"Moving model to {device}")
            encoder.to(device)

        z, y = extract_features(model=encoder, data_loader=data_loader, device=device)
        return self.fit_features(z, y)

    def fit_features(self, z: Tensor, labels: Tensor, batch_size: int = 4096) -> Self:
        """
        Build the energy-scaled feature bank from pre-extracted features, on the device of ``z``,
        and fit the search with it.

        :param z: features of shape :math:`N \\times D`
        :param labels: class labels of shape :math:`N`; OOD samples (label below zero) are ignored
        :param batch_size: number of samples processed at once; lower it to reduce peak GPU memory
        :return: self
        :raise ValueError: if fewer than :math:`k` ID samples are given
        """
        known = is_known(labels).to(z.device)
        _check_k(self.k, int(known.sum()), "ID samples")

        device = self.device or z.device
        z = z[known].detach()

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        def scaled(z_batch: Tensor) -> Tensor:
            energy = torch.logsumexp(self.head(z_batch), dim=1)
            return F.normalize(z_batch, dim=1) * energy[:, None]

        self.search.fit(apply_in_batches(scaled, z, batch_size, device))
        self.search.to(device)  # no-op if pinned
        self._is_fitted = True

        return self

    @torch.no_grad()
    def predict_features(self, z: Tensor) -> Tensor:
        """
        Compute the NNGuide outlier score from pre-extracted features.

        :param z: features of shape :math:`B \\times D`
        :return: outlier scores of shape :math:`B`
        :raise RequiresFittingException: if the detector has not been fitted
        :raise ValueError: if ``k`` exceeds the number of fitted samples
        """
        if not self._is_fitted:
            raise RequiresFittingException()
        _check_k(self.k, self.search.n_samples, "fitted samples")

        device = self.device or z.device
        z = z.to(device)

        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

        energy = torch.logsumexp(self.head(z), dim=1)
        similarities, _ = self.search.search(F.normalize(z, dim=1), self.k)
        guidance = similarities.mean(dim=1)

        # higher guidance * energy = more in-distribution, so negate for outlier score
        return -(guidance * energy)

    def to(self, device: Union[str, torch.device]) -> Self:
        """
        Move the detector to ``device``, including the bank unless the search is pinned.
        """
        super().to(device)
        self.search.to(device)
        return self
