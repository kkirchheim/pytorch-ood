"""

..  autoclass:: pytorch_ood.detector.WeightedEBO
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional

import torch
from torch import Tensor
from typing_extensions import Self

from ..api import DetectorInfo, LogitsDetector, Paper, Task


class WeightedEBO(LogitsDetector):
    """
    Implements the Weighted Energy Based Score of  *VOS: Learning what you don’t know by virtual outlier synthesis*.

    This method calculates the energy from the weighted logits. The energy :math:`E(x)` is returned as outlier score.
    The weights can be obtained, for example, by training with the :class:`pytorch_ood.loss.VOSRegLoss`.

    Overall, the score is defined as:

    .. math::
        E(x) = - \\log{\\sum_i w_{i} e^{f_i(x)}}

    where :math:`f_i(x)` indicates the :math:`i^{th}` logit value predicted by :math:`f` and :math:`w_i` indicates the weight of class :math:`i`, i.e. the ReLU of the given ``weights``.

    .. rubric:: Examples

    .. code-block:: python

        weights = torch.nn.Linear(num_classes, 1).weight
        detector = WeightedEBO(model, weights)
        scores = detector(images)
    """

    info = DetectorInfo(
        paper=Paper(
            title="VOS: Learning What You Don't Know by Virtual Outlier Synthesis",
            venue="ICLR",
            year=2022,
            url="https://arxiv.org/pdf/2202.01197.pdf",
            code="https://github.com/deeplearning-wisc/vos/",
        ),
        tasks={Task.CLASSIFICATION, Task.SEGMENTATION},
    )

    def __init__(self, model: Optional[torch.nn.Module], weights: torch.Tensor):
        """
        :param model: neural network :math:`f` to use, is assumed to output logits. Can be
            ``None`` when using ``predict_logits(...)`` directly.
        :param weights: tensor of shape :math:`C` or :math:`1 \\times C`, where :math:`C` is the number of classes
            (e.g. ``Linear(C, 1).weight``). Negative entries are clipped to 0.
        """
        super(WeightedEBO, self).__init__()

        self.model = model
        self.weights = weights

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits of shape :math:`B \\times C` (or :math:`B \\times C \\times H \\times W`)
        :return: outlier scores of shape :math:`B` (or :math:`B \\times H \\times W`)
        """
        return self.score(logits, self.weights)

    @staticmethod
    def score(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted energy of the logits, see :class:`WeightedEBO <pytorch_ood.detector.WeightedEBO>`.

        :param logits: logits of shape :math:`B \\times C` (or :math:`B \\times C \\times H \\times W`)
        :param weights: tensor with the class weights, of shape :math:`C` or :math:`1 \\times C`
        :return: energy of shape :math:`B` (or :math:`B \\times H \\times W`)
        :raise ValueError: if ``logits`` is neither 2-dimensional nor 4-dimensional
        """
        weights = weights.to(logits.device).relu()

        # Classification
        if len(logits.shape) == 2:
            energy = torch.log(torch.sum((weights * torch.exp(logits)), dim=1, keepdim=False))

            return -energy
        # Segmentation
        elif len(logits.shape) == 4:
            # Permutation depends on shape of logits

            logits = logits.permute(0, 2, 3, 1)

            energy = torch.log(
                torch.sum(
                    (weights * torch.exp(logits)),
                    dim=3,
                    keepdim=False,
                )
            )

            return -energy
        else:
            raise ValueError(f"Unsupported input shape: {logits.shape}")
