# Parts of this code are taken from https://github.com/Jingkang50/OpenOOD/blob/main/openood/postprocessors/gram_postprocessor.py
"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Gram
    :members:
    :exclude-members: fit_features
"""
import logging
from typing import Optional, TypeVar, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np
from ..api import Detector, ModelNotSetException, RequiresFittingException

import torch.nn.functional as F

log = logging.getLogger(__name__)

Self = TypeVar("Self")


class Gram(Detector):
    """
    Implements the on Gram matrices based Method from the paper *Detecting Out-of-Distribution Examples with
    In-distribution Examples and Gram Matrices*.

    The Gram detector identifies OOD examples by analyzing feature correlations within the layers of a neural network using Gram matrices,
    which are computed as:

    .. math :: G^p_l = \\left(F_l^p F_l^{p \\top}\\right)^{\\frac{1}{p}}

    Where :math:`F_l` is the feature-map in layer :math:`l`.
    The Gram matrices capture the pairwise correlations between feature maps, which can be seen as capturing the image style.
    For each layer, matrices for several values of :math:`p`, called *''poles''* are computed.
    During training, class-specific minimum and maximum bounds are calculated for each entry in the Gram matrices
    of the ID data in multiple layers of a neural network.
    For a test input :math:`x`, deviations are calculated layer-wise by comparing the Gram matrix values against the stored bounds.
    The total deviation across all layers :math:`l` is normalized using the expected deviation for that layer:

    .. math :: \\Delta(x) = \\sum_{l} \\frac{\\delta_l(x)}{\\mathbb{E}[\\delta_l]}


    :see Implementation: `GitHub <https://github.com/VectorInstitute/gram-ood-detection>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1912.12510>`__
    """

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
        self.feature_min, self.feature_max = None, None

    @torch.no_grad()
    def _create_feature_list(self, data: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        :param data: input tensor
        :return: feature list
        """
        feature_list = []
        # data_tmp = data.clone()
        for idx in range(self.num_layer):
            data = self.feature_layers[idx](data)
            feature_list.append(data.clone())

        logits = self.head(data)

        assert (
            logits.shape[1] == self.num_classes
        ), f"You set num_classes={self.num_classes} but got {logits.shape[1]}"

        return logits, feature_list

    def fit(self: Self, data_loader: DataLoader, device: str = None) -> Self:
        """
        Calculate the minimum and maximum values for the Gram matrices of the training data.

        :param data_loader: data loader for training data
        :param device: device to run the model on

        :return: self
        """
        num_poles = len(self.num_poles_list)
        feature_class = [
            [[None for x in range(num_poles)] for y in range(self.num_layer)]
            for z in range(self.num_classes)
        ]

        mins = [
            [[None for x in range(num_poles)] for y in range(self.num_layer)]
            for z in range(self.num_classes)
        ]
        maxs = [
            [[None for x in range(num_poles)] for y in range(self.num_layer)]
            for z in range(self.num_classes)
        ]

        with torch.no_grad():
            # collect features and compute gram matrix
            for n, (x, y) in enumerate(data_loader):
                data = x.to(device)
                label = y.to(device)
                _, feature_list = self._create_feature_list(data)
                label_list = label.tolist()
                for layer_idx in range(self.num_layer):

                    for pole_idx, p in enumerate(self.num_poles_list):
                        temp = feature_list[layer_idx].detach()

                        temp = temp**p
                        temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
                        temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
                        temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(
                            temp.shape[0], -1
                        )

                        temp = temp.tolist()
                        for feature, label in zip(temp, label_list):
                            if isinstance(feature_class[label][layer_idx][pole_idx], type(None)):
                                feature_class[label][layer_idx][pole_idx] = feature
                            else:
                                feature_class[label][layer_idx][pole_idx].extend(feature)

                if n % 100 == 0:
                    log.debug(f"Fitting: {n}/{len(data_loader)}")

            for label in range(self.num_classes):
                for layer_idx in range(self.num_layer):
                    for poles_idx in range(num_poles):
                        feature = torch.tensor(
                            np.array(feature_class[label][layer_idx][poles_idx])
                        )
                        current_min = feature.min(dim=0, keepdim=True)[0]
                        current_max = feature.max(dim=0, keepdim=True)[0]

                        if mins[label][layer_idx][poles_idx] is None:
                            mins[label][layer_idx][poles_idx] = current_min
                            maxs[label][layer_idx][poles_idx] = current_max
                        else:
                            mins[label][layer_idx][poles_idx] = torch.min(
                                current_min, mins[label][layer_idx][poles_idx]
                            )
                            maxs[label][layer_idx][poles_idx] = torch.max(
                                current_min, maxs[label][layer_idx][poles_idx]
                            )
            self.feature_min = torch.tensor(mins)
            self.feature_max = torch.tensor(maxs)
            return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        raise NotImplementedError("This method is not implemented. Use fit instead.")

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

        logits, feature_list = self._create_feature_list(x)

        return self._score(logits, feature_list)

    def predict_features(self, logits: Tensor, feature_list: List[Tensor]) -> Tensor:
        """
        :param logits: logits given by your model
        :param feature_list: list of features extracted from the model
        :return: Gram based Deviations
        """
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

        deviations = torch.zeros(size=(logits.shape[0],), device=device)

        predictions = torch.argmax(logits, dim=1)

        self.feature_min = self.feature_min.to(device)
        self.feature_max = self.feature_max.to(device)

        feature_min_prep = (self.feature_min + 10**-6).abs()
        feature_max_prep = (self.feature_max + 10**-6).abs()

        # compute sample level deviation
        for layer_idx in range(self.num_layer):
            for pole_idx, p in enumerate(self.num_poles_list):
                # get gram matrix
                temp = feature_list[layer_idx].to(device)
                temp = temp**p
                temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
                temp = (torch.matmul(temp, temp.transpose(dim0=2, dim1=1))).sum(dim=2)
                temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

                temp_sums = temp.sum(dim=1)

                min_norm = feature_min_prep[predictions, layer_idx, pole_idx]
                max_norm = feature_max_prep[predictions, layer_idx, pole_idx]

                features_min = self.feature_min[predictions, layer_idx, pole_idx]
                features_max = self.feature_max[predictions, layer_idx, pole_idx]

                # compute the deviations with train data
                deviations += F.relu(features_min - temp_sums) / min_norm
                deviations += F.relu(temp_sums - features_max) / max_norm

        return -deviations / 50
