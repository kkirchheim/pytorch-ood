"""
..  autoclass:: pytorch_ood.detector.KLMatching
    :members:

"""
import logging
from typing import TypeVar

import torch
from torch.nn import Parameter, ParameterDict
from torch.utils.data import DataLoader

from pytorch_ood.utils import TensorBuffer, is_known

from ..api import Detector, RequiresFittingException

log = logging.getLogger()

Self = TypeVar("Self")


class KLMatching(Detector):
    """
    Implements KL-Matching from the paper *Scaling Out-of-Distribution Detection for Real-World Settings*.

    For each class, an typical posterior distribution
    :math:`d_y = \\mathbb{E}_{x \\sim \\mathcal{X}_{val}}[p(y \\vert x)]` is
    estimated, where :math:`y` is the class with the maximum posterior  :math:`y = \\arg\\max_y p(y \\vert x)`,
    as predicted by the model. Note that the method does not require class labels for the validation set.
    During evaluation, the KL-Divergence between the observed and the typical posterior
    :math:`D_{KL}[p(y \\vert x) \\Vert d_y]` is used as outlier score.

    This method can also be applied to multi-class settings.

    :see Paper: `ArXiv <https://arxiv.org/abs/1911.11132>`__
    """

    def __init__(self, model: torch.nn.Module):
        """
        :param model: neural network, is assumed to output logits.
        """
        super(KLMatching, self).__init__()
        self.model = model
        self.dists: ParameterDict = ParameterDict()  #: Typical posteriors per class

    def fit(self: Self, data_loader: DataLoader, device="cpu") -> Self:
        """
        Estimates typical distributions for each class.
        Ignores OOD samples.

        :param data_loader: validation data loader
        :param device: device which should be used for calculations
        """
        buffer = TensorBuffer()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                known = is_known(y)
                logits = self.model(x[known])
                y_hat = logits.argmax(dim=1)
                buffer.append("logits", logits)
                buffer.append("label", y_hat)

        probabilities = buffer.get("logits").softmax(dim=1)
        labels = buffer.get("label")

        for label in labels.unique():
            log.debug(f"Fitting class {label}")
            d_k = probabilities[labels == label].mean(dim=0)
            self.dists[str(label.item())] = Parameter(d_k)

        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates KL-Divergence between predicted posteriors and typical posteriors.

        :param x: input tensor, will be passed through model
        :return: Outlier scores
        """
        if len(self.dists) == 0:
            raise RequiresFittingException("KL-Matching has to be fitted on validation data.")

        #
        device = x.device
        self.dists.to(device)

        probs = self.model(x).softmax(dim=1)
        predictions = probs.argmax(dim=1)
        scores = torch.empty(size=(probs.shape[0],), device=device)

        for label in predictions.unique():
            if str(label.item()) not in self.dists:
                raise ValueError(f"Label {label.item()} not fitted.")

            dist = self.dists[str(label.item())]
            class_p = probs[predictions == label]
            class_d = dist.unsqueeze(0).repeat(class_p.shape[0], 1)
            d_kl = (class_p * (class_p / class_d).log()).sum(dim=1)
            scores[predictions == label] = d_kl

        return scores
