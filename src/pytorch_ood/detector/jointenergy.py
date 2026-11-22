"""

..  autoclass:: pytorch_ood.detector.JointEnergy
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: fit, fit_logits
"""

from typing import Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import softplus

from ..api import DetectorInfo, LogitsDetector, Paper, Task


class JointEnergy(LogitsDetector):
    """
    Implements the JointEnergy score of *Can multi-label classification networks know what
    they don't know?* for multi-label classifiers, i.e. models trained with one sigmoid output
    per label.

    Each label :math:`i` has the energy :math:`E_i(x) = -\\log(1 + e^{f_i(x)})`. The paper sums
    the negative label-wise energies into an in-distribution score:

    .. math::
        E_{joint}(x) = \\sum_{i=1}^{C} \\log(1 + e^{f_i(x)})

    where :math:`f_i(x)` is the :math:`i^{th}` logit. This detector returns
    :math:`-E_{joint}(x)`, so larger values indicate OOD.

    The score assumes independent sigmoid outputs; for a softmax classifier, use
    :class:`EnergyBased` instead.
    """

    info = DetectorInfo(
        paper=Paper(
            title="Can multi-label classification networks know what they don't know?",
            venue="NeurIPS",
            year=2021,
            url="https://arxiv.org/abs/2109.14162",
            code="https://github.com/deeplearning-wisc/multi-label-ood",
        ),
        tasks={Task.CLASSIFICATION},
        ai_coded=True,
    )

    def __init__(self, model: Optional[Module]):
        """
        :param model: neural network to use. Can be ``None`` when using
            ``predict_logits(...)`` directly.
        """
        super(JointEnergy, self).__init__()
        self.model = model

    def predict_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: logits given by the model, shape :math:`B \\times C`
        :return: outlier scores of shape :math:`B`
        """
        return JointEnergy.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits of input, shape :math:`B \\times C`
        :return: negative joint energy, shape :math:`B`
        """
        # softplus(x) = log(1 + exp(x)), computed without overflow for large logits
        return -softplus(logits).sum(dim=1)
