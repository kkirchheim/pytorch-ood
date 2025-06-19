"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Gradient
    :members:
    :exclude-members: fit_features, predict_features, fit
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing import TypeVar

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")


class Gradient(Detector):
    """
    Detector from the paper *Gradients as a Measure of Uncertainty in Neural Networks*.

    For each input sample, computes the binary cross-entropy loss between logits and a "confounding label",
    which is a vector of all ones. Then, for each set of parameters in the model (as given
    by ``model.parameters()``), computes up the squared :math:`\\ell_2`-norm of the
    gradients of the loss w.r.t. that parameter. The outlier score is the sum of these squared norms,
    which is the sum of all squared gradients.

    The idea is that higher gradient norms indicates that the model would require large
    parameter updates to accommodate the input, i.e., it is less familiar or
    more uncertain, and hence more likely to be OOD.

    :see Paper: `ICIP <https://arxiv.org/abs/2008.08030v2>`__
    """

    def __init__(self, model: torch.nn.Module):
        """
        :param model: A pre-trained classification model
        """
        if model is None:
            raise ModelNotSetException("Model must be provided.")

        self.model = model

    def fit(self, data_loader: DataLoader, **kwargs) -> Self:
        return self

    def fit_features(self, x: Tensor, y: Tensor) -> Self:
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Compute outlier scores from input batch.

        We will use the device of the model parameters for computations.

        :param x: input, will be passed through network
        :return: vector of outlier scores
        """
        if self.model is None:
            raise ModelNotSetException()

        device = next(self.model.parameters()).device
        x = x.to(device)
        scores = []

        for xi in x:
            with torch.enable_grad():
                self.model.zero_grad()
                logits = self.model(xi.unsqueeze(0))
                y_conf = torch.ones_like(logits)
                loss = F.binary_cross_entropy(logits.softmax(dim=1), y_conf, reduction="mean")
                loss.backward()

                # Sum of squared L2 norms over all gradients
                total_norm = 0.0
                for name, p in self.model.named_parameters():

                    if p.grad is not None:
                        total_norm += torch.sum(p.grad.detach() ** 2)
                scores.append(total_norm)

        return torch.stack(scores)

    def predict_features(self, x: Tensor) -> Tensor:
        """
        This is not possible, as we have to compute a backward pass through the model.
        """
        raise NotImplementedError
