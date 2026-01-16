"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.GradNorm
    :members:
    :exclude-members: fit_features, predict_features, fit
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing import TypeVar, Callable

from ..api import Detector, ModelNotSetException

Self = TypeVar("Self")


class GradNorm(Detector):
    """
    Detector from the paper *Gradients as a Measure of Uncertainty in Neural Networks*.

    For each input sample, computes the binary cross-entropy loss between logits and a "confounding label",
    which is a vector of all ones. Then, for each set of parameters in the model (as given
    by ``model.named_parameters()``), computes up the squared :math:`\\ell_2`-norm of the
    gradients of the loss w.r.t. that parameter. The outlier score is the sum of these squared norms.

    The idea is that higher gradient norms indicates that the model would require large
    parameter updates to accommodate the input, i.e., for such data, it is less familiar or
    more uncertain, and hence more likely to be OOD.

    .. note:: OpenOOD uses only the gradients of the final classification head, which
     makes this computationally cheaper. You can achieve something similar by setting ``param_filter``. Still, this
     method will compute gradients for all parameters unless you explicitly deactivate
     gradient calculation for parameters. For an example, see :doc:`here <auto_examples/detectors/gradnorm>`

    :see Paper: `ICIP <https://arxiv.org/abs/2008.08030v2>`__
    """

    def __init__(self, model: torch.nn.Module, param_filter: Callable[[str], bool] = None):
        """
        :param model: A pre-trained classification model
        :param param_filter: Function which indicates whether a named parameter should be included in the scoring. If none
            give, all parameters will be used.
        """
        if model is None:
            raise ModelNotSetException("Model must be provided.")

        def default_filter(x):
            return True

        self.param_filter = param_filter or default_filter

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
                y_conf = torch.ones_like(logits, device=device)
                loss = F.binary_cross_entropy(logits.softmax(dim=1), y_conf, reduction="sum")
                loss.backward()

                # Sum of squared L2 norms over all gradients
                total_norm = torch.tensor(0.0, device=device)
                for name, p in self.model.named_parameters():

                    if self.param_filter(name) and p.grad is not None:
                        total_norm += torch.sum(p.grad.detach() ** 2)

                scores.append(total_norm)

        return torch.stack(scores)

    def predict_features(self, x: Tensor) -> Tensor:
        """
        This is not possible, as we have to compute a backward pass through the model.
        """
        raise NotImplementedError
