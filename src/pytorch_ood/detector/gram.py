# Adapted from https://github.com/VectorInstitute/gram-ood-detection
"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Gram
    :members:
    :inherited-members:
    :show-inheritance:
"""

import logging
from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from ..api import ModelNotSetException, RequiresFittingException, StructuredDetector

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class Gram(StructuredDetector):
    """
    Implements the on Gram matrices based Method from the paper *Detecting Out-of-Distribution Examples with
    In-distribution Examples and Gram Matrices*.

    The Gram detector identifies OOD examples by analyzing feature correlations within the layers of a neural network using Gram matrices,
    which are computed as:

    .. math :: G^p_l = \\left(F_l^p F_l^{p \\top}\\right)^{\\frac{1}{p}}

    Where :math:`F_l` is the feature-map in layer :math:`l`.
    The Gram matrices capture the pairwise correlations between feature maps, which can be seen as capturing the image style.
    For each layer, matrices for several values of :math:`p`, called *''poles''* are computed.
    During fitting, class-specific minimum and maximum bounds are calculated for each entry of the
    (row-summed) Gram matrices of the ID data in multiple layers of the neural network.
    For a test input :math:`x`, deviations :math:`\\delta_l(x)` are calculated layer-wise by comparing each Gram
    matrix entry against the stored bounds of the predicted class.
    The total deviation is the sum over all layers :math:`l`, normalized by the expected deviation of the layer,
    which is estimated on a held-out fraction of the fitting data:

    .. math :: \\Delta(x) = \\sum_{l} \\frac{\\delta_l(x)}{\\mathbb{E}[\\delta_l]}

    Higher values indicate more likely OOD inputs.

    :see Implementation: `GitHub <https://github.com/VectorInstitute/gram-ood-detection>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1912.12510>`__
    """

    requires_fit = True

    #: fraction of the fitting data (per class) held out to estimate the expected deviation
    validation_fraction = 0.1

    def __init__(
        self,
        head: Module,
        feature_layers: List[Module],
        num_classes: int,
        num_poles_list: List[int] = None,
    ):
        """
        :param head: the head of the model
        :param feature_layers: the layers of the model to be used for feature extraction
        :param num_classes: the number of classes in the dataset
        :param num_poles_list: the list of poles to be used for higher-order Gram matrices
        """
        super(Gram, self).__init__()
        self.head = head
        self.feature_layers = feature_layers
        self.num_layer = len(feature_layers)
        self.num_classes = num_classes
        if num_poles_list is None:
            self.num_poles_list = range(1, len(self.feature_layers) + 1)
        else:
            self.num_poles_list = num_poles_list

        #: per-entry lower bounds, indexed ``[layer][pole]``, each of shape ``(num_classes, C_l)``
        self.feature_min: Optional[List[List[Tensor]]] = None
        #: per-entry upper bounds, indexed ``[layer][pole]``, each of shape ``(num_classes, C_l)``
        self.feature_max: Optional[List[List[Tensor]]] = None
        #: expected deviation :math:`\mathbb{E}[\delta_l]` per layer, shape ``(num_layer,)``
        self.layer_norm: Optional[Tensor] = None

    @torch.no_grad()
    def _create_feature_list(self, data: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        :param data: input tensor
        :return: logits and features for each layer
        """
        feature_list = []
        for idx in range(self.num_layer):
            data = self.feature_layers[idx](data)
            feature_list.append(data.clone())

        logits = self.head(data)

        assert logits.shape[1] == self.num_classes, (
            f"You set num_classes={self.num_classes} but got {logits.shape[1]}"
        )

        return logits, feature_list

    @staticmethod
    def _gram_vector(feature: Tensor, p: int) -> Tensor:
        """
        Row sums of the :math:`p`-th order Gram matrix of a batch of feature maps.

        :param feature: feature maps of shape :math:`(B, C, ...)`
        :return: gram statistics of shape :math:`(B, C)`
        """
        temp = feature.detach() ** p
        temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
        temp = torch.matmul(temp, temp.transpose(dim0=2, dim1=1)).sum(dim=2)
        return (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    @staticmethod
    def _deviation(g: Tensor, mins: Tensor, maxs: Tensor) -> Tensor:
        """
        Elementwise out-of-bounds deviation, summed over gram entries.

        :param g: gram statistics of shape :math:`(B, C)`
        :param mins: lower bounds, broadcastable to :math:`(B, C)`
        :param maxs: upper bounds, broadcastable to :math:`(B, C)`
        :return: deviations of shape :math:`(B,)`
        """
        dev = (F.relu(mins - g) / torch.abs(mins + 1e-6)).sum(dim=1)
        dev = dev + (F.relu(g - maxs) / torch.abs(maxs + 1e-6)).sum(dim=1)
        return dev

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Calculate the per-entry minimum and maximum bounds of the Gram matrix statistics of
        the training data, as well as the expected deviation per layer. Ignores OOD inputs.

        :param data_loader: data loader for training data
        :return: self
        """
        device = self.device
        if device is None:
            device = "cpu"
            log.warning(f"No device set. Will use '{device}'.")
            self.to(device)

        num_poles = len(self.num_poles_list)

        # collected[class][layer][pole] -> list of (B_i, C_l) gram statistics
        collected = [
            [[[] for _ in range(num_poles)] for _ in range(self.num_layer)]
            for _ in range(self.num_classes)
        ]

        with torch.no_grad():
            for n, (x, y) in enumerate(data_loader):
                _, feature_list = self._create_feature_list(x.to(device))

                for layer_idx in range(self.num_layer):
                    for pole_idx, p in enumerate(self.num_poles_list):
                        g = self._gram_vector(feature_list[layer_idx], p).cpu()
                        for clazz in y.unique():
                            c = int(clazz.item())
                            if c < 0:
                                continue
                            collected[c][layer_idx][pole_idx].append(g[y == clazz])

                if n % 100 == 0:
                    log.debug(f"Fitting: {n}/{len(data_loader)}")

        # hold out a fraction of each class to estimate the expected deviation per layer
        splits = {}
        for c in range(self.num_classes):
            if not collected[c][0][0]:
                raise ValueError(f"No ID samples for class {c}")
            n_c = sum(t.shape[0] for t in collected[c][0][0])
            perm = torch.randperm(n_c)
            n_val = int(self.validation_fraction * n_c)
            splits[c] = (perm[n_val:], perm[:n_val])

        feature_min = [[None] * num_poles for _ in range(self.num_layer)]
        feature_max = [[None] * num_poles for _ in range(self.num_layer)]
        val_deviations = [[] for _ in range(self.num_layer)]

        for layer_idx in range(self.num_layer):
            for pole_idx in range(num_poles):
                cls_min, cls_max = [], []
                for c in range(self.num_classes):
                    g = torch.cat(collected[c][layer_idx][pole_idx])
                    trn, _ = splits[c]
                    cls_min.append(g[trn].min(dim=0).values)
                    cls_max.append(g[trn].max(dim=0).values)
                feature_min[layer_idx][pole_idx] = torch.stack(cls_min)
                feature_max[layer_idx][pole_idx] = torch.stack(cls_max)

            for c in range(self.num_classes):
                trn, val = splits[c]
                if len(val) == 0:
                    continue
                dev = torch.zeros(len(val))
                for pole_idx in range(num_poles):
                    g_val = torch.cat(collected[c][layer_idx][pole_idx])[val]
                    dev += self._deviation(
                        g_val,
                        feature_min[layer_idx][pole_idx][c],
                        feature_max[layer_idx][pole_idx][c],
                    )
                val_deviations[layer_idx].append(dev)

        norms = []
        for layer_idx in range(self.num_layer):
            expected = torch.tensor(1.0)
            if val_deviations[layer_idx]:
                mean_dev = torch.cat(val_deviations[layer_idx]).mean()
                if mean_dev > 0:
                    expected = mean_dev
            norms.append(expected)

        self.feature_min = feature_min
        self.feature_max = feature_max
        self.layer_norm = torch.stack(norms)
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate deviation for inputs

        :param x: input tensor, will be passed through model

        :return: Gram based deviations
        """
        if self.head is None:
            raise ModelNotSetException

        if self.feature_min is None:
            raise RequiresFittingException

        device = self.device or x.device
        x = x.to(device)
        logits, feature_list = self._create_feature_list(x)

        return self._score(logits, feature_list)

    def predict_structured(self, logits: Tensor, feature_list: List[Tensor]) -> Tensor:
        """
        :param logits: logits given by your model
        :param feature_list: list of features extracted from the model
        :return: Gram based Deviations
        """
        device = self.device or logits.device
        logits = logits.to(device)
        feature_list = [f.to(device) for f in feature_list]
        return self._score(logits, feature_list)

    @torch.no_grad()
    def _score(self, logits: Tensor, feature_list: List[Tensor]) -> Tensor:
        """
        Calculate deviation for inputs

        :param logits: logits of input
        :param feature_list: list of features extracted from the model

        :return: Gram based deviations
        """
        if self.feature_min is None or self.feature_max is None:
            raise RequiresFittingException("Fit the detector first.")

        device = logits.device
        predictions = torch.argmax(logits, dim=1)
        deviations = torch.zeros(size=(logits.shape[0],), device=device)

        for layer_idx in range(self.num_layer):
            layer_dev = torch.zeros(size=(logits.shape[0],), device=device)
            for pole_idx, p in enumerate(self.num_poles_list):
                g = self._gram_vector(feature_list[layer_idx].to(device), p)
                mins = self.feature_min[layer_idx][pole_idx].to(device)[predictions]
                maxs = self.feature_max[layer_idx][pole_idx].to(device)[predictions]
                layer_dev += self._deviation(g, mins, maxs)

            deviations += layer_dev / self.layer_norm[layer_idx].to(device)

        return deviations
